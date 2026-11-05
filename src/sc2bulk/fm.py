"""Frozen single-cell foundation models: Geneformer V2-104M and scGPT whole-human.
Each follows its model's
published embedding path; the deviations (which genes are in the input; scGPT's two random
steps seeded per profile) are stated in the spec and Methods. Imported only in the FM
environments (stage 07). Labels are never read.

Inputs are raw counts on a gene axis (the pair's shared genes, or a unit's panel). Both models'
encodings are invariant to the library-size normalization (Geneformer ranks count / median
within a profile; scGPT bins within-profile quantiles), so counts go in directly.
"""
import json
import pickle

import numpy as np
import scipy.sparse as sp
import torch

from sc2bulk import seeds

GENEFORMER_SPECIAL = 2   # <cls> and <eos> (V2 models)
SCGPT_MAX_LENGTH = 1200  # including <cls>
SCGPT_BINS = 51


def collapse(X, targets: np.ndarray):
    """Sum gene columns that map to the same target (Geneformer's collapse_gene_ids); columns
    with target -1 are dropped. Returns (cells x unique targets, float32; the unique targets)."""
    keep = np.flatnonzero(targets >= 0)
    unique, column = np.unique(targets[keep], return_inverse=True)
    assign = sp.csr_matrix((np.ones(len(keep), dtype=np.float32), (keep, column)), shape=(len(targets), len(unique)))
    return sp.csr_matrix(X, dtype=np.float32) @ assign, unique


class Geneformer:
    """Geneformer V2: each profile's nonzero genes ranked by count over the gene's corpus median
    (np.argsort(-v), as the tokenizer does), truncated to the input size minus <cls>/<eos>; the
    embedding is the <cls> output at hidden layer n_layers - 1 (EmbExtractor's defaults for V2:
    emb_mode "cls", emb_layer -1). float32, SDPA attention."""

    def __init__(self, repo, model_dir, device="cuda", batch_size=16):
        from transformers import BertForMaskedLM

        package = repo / "geneformer"
        load = lambda name: pickle.load(open(package / name, "rb"))
        self.token = load("token_dictionary_gc104M.pkl")
        self.median = load("gene_median_dictionary_gc104M.pkl")
        self.mapping = load("ensembl_mapping_dict_gc104M.pkl")    # Ensembl id -> collapsed Ensembl id
        self.name_to_id = load("gene_name_id_dict_gc104M.pkl")    # human symbol -> Ensembl id
        self.model = BertForMaskedLM.from_pretrained(model_dir, output_hidden_states=True,
                                                     attn_implementation="sdpa").to(device).eval()
        self.input_size = int(self.model.config.max_position_embeddings)
        self.layer = int(self.model.config.num_hidden_layers) - 1
        self.device, self.batch_size = device, batch_size

    def targets(self, ensembl_ids, symbols) -> np.ndarray:
        """Each gene's token, or -1 if unreadable. A gene's id(s) (';'-joined) pass through the
        collapsing dictionary and must reach exactly one token with a median; a gene with no id
        (human pairs) is looked up by symbol in the gc104M name dictionary."""
        out = []
        for ids, symbol in zip(ensembl_ids, symbols):
            candidates = ids.split(";") if ids else ([self.name_to_id[symbol]] if symbol in self.name_to_id else [])
            # ids missing from the collapsing dictionary are dropped, as the tokenizer's .map() does
            reached = {self.mapping.get(i) for i in candidates} - {None}
            reached = {e for e in reached if e in self.token and e in self.median}
            out.append(self.token[reached.pop()] if len(reached) == 1 else -1)
        return np.array(out)

    def _encode(self, row: np.ndarray, tokens: np.ndarray, medians: np.ndarray):
        v = row / medians
        nz = np.flatnonzero(v > 0)
        order = nz[np.argsort(-v[nz])]
        room = self.input_size - GENEFORMER_SPECIAL
        ids = np.concatenate([[self.token["<cls>"]], tokens[order[:room]], [self.token["<eos>"]]]).astype(np.int32)
        return ids, max(0, len(order) - room)

    @torch.no_grad()
    def embed(self, X, tokens: np.ndarray):
        """(embeddings, genes truncated per profile) for profiles X (rows; columns = the
        collapsed tokens)."""
        token_to_id = {t: e for e, t in self.token.items()}
        medians = np.array([self.median[token_to_id[t]] for t in tokens])
        X = sp.csr_matrix(X)
        encoded = [self._encode(X[i].toarray().ravel(), tokens, medians) for i in range(X.shape[0])]
        order = np.argsort([len(ids) for ids, _ in encoded], kind="stable")   # fixed, length-sorted batches
        out = np.empty((X.shape[0], self.model.config.hidden_size), dtype=np.float32)
        pad = self.token["<pad>"]
        for start in range(0, len(order), self.batch_size):
            batch = order[start:start + self.batch_size]
            length = max(len(encoded[i][0]) for i in batch)
            ids = torch.full((len(batch), length), pad, dtype=torch.long)
            for row, i in enumerate(batch):
                ids[row, :len(encoded[i][0])] = torch.tensor(encoded[i][0])
            mask = (ids != pad).long()
            # the encoder only (BertForMaskedLM's 20,275-way logits aren't needed; hidden states are identical)
            hidden = self.model.bert(input_ids=ids.to(self.device), attention_mask=mask.to(self.device)).hidden_states
            out[batch] = hidden[self.layer][:, 0, :].float().cpu().numpy()
        return out, np.array([d for _, d in encoded])


class ScGPT:
    """scGPT whole-human, run as scgpt.tasks.cell_emb.get_batch_cell_embeddings does (cls mode):
    a profile's nonzero vocabulary genes, values binned into 51 within-profile quantile bins, a
    random 1,199 of them when more are expressed, the <cls> output L2-normalized, under
    autocast. Both random steps (binning's tie-break, the gene sample) are seeded per profile
    (np.random.seed and torch.manual_seed with a seed keyed by the profile id), so an
    embedding doesn't depend on batch composition. use_fast_transformer=False (disclosed)."""

    def __init__(self, model_dir, device="cuda", batch_size=64):
        from scgpt.model import TransformerModel
        from scgpt.tokenizer import GeneVocab
        from scgpt.utils import load_pretrained

        self.configs = json.load(open(model_dir / "args.json"))
        assert self.configs["embsize"] == 512, "args.json is not the whole-human checkpoint's (embsize 512)"
        self.vocab = GeneVocab.from_file(model_dir / "vocab.json")
        for special in ("<pad>", "<cls>", "<eoc>"):
            if special not in self.vocab:
                self.vocab.append_token(special)
        self.vocab.set_default_index(self.vocab["<pad>"])
        c = self.configs
        self.model = TransformerModel(
            ntoken=len(self.vocab), d_model=c["embsize"], nhead=c["nheads"], d_hid=c["d_hid"],
            nlayers=c["nlayers"], nlayers_cls=c["n_layers_cls"], n_cls=1, vocab=self.vocab,
            dropout=c["dropout"], pad_token=c["pad_token"], pad_value=c["pad_value"], do_mvc=True,
            do_dab=False, use_batch_labels=False, domain_spec_batchnorm=False, explicit_zero_prob=False,
            use_fast_transformer=False, pre_norm=False)
        load_pretrained(self.model, torch.load(model_dir / "best_model.pt", map_location=device), verbose=False)
        self.model.to(device).eval()
        self.device, self.batch_size = device, batch_size

    def targets(self, symbols) -> np.ndarray:
        """Each gene's vocabulary id, or -1 if its symbol isn't in the vocabulary."""
        return np.array([self.vocab[s] if s in self.vocab else -1 for s in symbols])

    def _collate_one(self, row: np.ndarray, gene_ids: np.ndarray, seed: int):
        from scgpt.data_collator import DataCollator

        np.random.seed(seed)
        torch.manual_seed(seed)
        nz = np.flatnonzero(row)
        example = {"id": 0, "genes": torch.from_numpy(np.insert(gene_ids[nz], 0, self.vocab["<cls>"])).long(),
                   "expressions": torch.from_numpy(np.insert(row[nz], 0, self.configs["pad_value"])).float()}
        collator = DataCollator(do_padding=True, pad_token_id=self.vocab[self.configs["pad_token"]],
                                pad_value=self.configs["pad_value"], do_mlm=False, do_binning=True,
                                max_length=SCGPT_MAX_LENGTH, sampling=True, keep_first_n_tokens=1)
        batch = collator([example])
        return batch["gene"][0], batch["expr"][0], max(0, len(nz) - (SCGPT_MAX_LENGTH - 1))

    @torch.no_grad()
    def embed(self, X, gene_ids: np.ndarray, profile_seeds: np.ndarray):
        """(embeddings, genes dropped by sampling per profile) for profiles X (rows; columns =
        the readable genes, in gene_ids order); profile_seeds keys each profile's random steps."""
        X = sp.csr_matrix(X)
        collated = [self._collate_one(X[i].toarray().ravel().astype(np.float32), gene_ids, int(profile_seeds[i]))
                    for i in range(X.shape[0])]
        pad_id, pad_value = self.vocab[self.configs["pad_token"]], self.configs["pad_value"]
        out = np.empty((X.shape[0], self.configs["embsize"]), dtype=np.float32)
        by_length = np.argsort([len(g) for g, _, _ in collated], kind="stable")   # less padding; fixed order
        for start in range(0, X.shape[0], self.batch_size):
            rows = by_length[start:start + self.batch_size]
            part = [collated[i] for i in rows]
            length = max(len(g) for g, _, _ in part)
            genes = torch.full((len(part), length), pad_id, dtype=torch.long)
            expr = torch.full((len(part), length), float(pad_value))
            for row, (g, e, _) in enumerate(part):
                genes[row, :len(g)], expr[row, :len(e)] = g, e
            genes = genes.to(self.device)
            with torch.cuda.amp.autocast(enabled=str(self.device).startswith("cuda")):
                emb = self.model._encode(genes, expr.to(self.device), src_key_padding_mask=genes.eq(pad_id))[:, 0, :]
            out[rows] = emb.float().cpu().numpy()
        out /= np.linalg.norm(out, axis=1, keepdims=True)
        return out, np.array([d for _, _, d in collated])


def profile_seeds(pair: str, profile_ids, feature_set_id: int) -> np.ndarray:
    """One seed per profile for scGPT's random steps, keyed by the profile id (not its position).
    A sample's pseudobulk, one-type pseudobulk and (where ids coincide) bulk profiles share a seed;
    harmless (different profiles), stated here."""
    from sc2bulk import preprocess

    base = seeds.derived("fm_sampling", pair, feature_set_id)
    return np.array([np.random.SeedSequence([base, preprocess.sample_seed(pair, p) % 2**32]).generate_state(1)[0]
                     for p in profile_ids], dtype=np.uint32)

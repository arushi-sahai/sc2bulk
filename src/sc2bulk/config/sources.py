"""Every raw input file: where it comes from and its SHA-256 checksum. Stage 01 downloads
the files it can and verifies every file against its checksum, so a changed or corrupted
source stops the pipeline instead of silently changing results.

Files listed under MANUAL can't be fetched by script (a browser-only portal, or a login);
stage 01 checks that they are in place and match, and otherwise stops with the
instructions given here.
"""

from pathlib import Path

import pandas as pd

GEO_SAMPLE = "https://ftp.ncbi.nlm.nih.gov/geo/samples/{prefix}nnn/{gsm}/suppl/{name}"
GEO_SERIES = "https://ftp.ncbi.nlm.nih.gov/geo/series/{prefix}nnn/{gse}/{kind}/{name}"


def geo_sample_url(name):
    gsm = name.split("_")[0]
    return GEO_SAMPLE.format(prefix=gsm[:-3], gsm=gsm, name=name)


def geo_series_url(gse, kind, name):
    return GEO_SERIES.format(prefix=gse[:-3], gse=gse, kind=kind, name=name)


# Hammond et al. 2019, GEO GSE121654: one genes x cells raw-count file per sample. All 47
# are downloaded; stage 02 drops the injury experiment (LPC and its SALINE controls).
HAMMOND_FILES = {
    "GSM3442006_E14_F_B10.dge.txt.gz": "9da49a158f626058f9c651a33c9c9cc80a21522d252f370b256d06ee346cfdeb",
    "GSM3442007_E14_M_B11.dge.txt.gz": "45093274c3f669465a61a5797c1099505491d003531246fa087b828b87b68d3b",
    "GSM3442008_E14_F_B12.dge.txt.gz": "97c474cd94845bb800e499c5f8bdeb429918322eda98002ee1d57943d1f3d19c",
    "GSM3442009_E14_F_C1.dge.txt.gz": "c83314888e5720f6b40ef4348588c5e6057866960d9210199d7987d558568cb3",
    "GSM3442010_E14_M_B9.dge.txt.gz": "8eac0808140c725bdf593e646f47485d5da995f173e6cddcc028b2f0951f99bb",
    "GSM3442011_E14_F_B6.dge.txt.gz": "a56b7bb31ff2404ddd8bf7ce8dd7c25636409015ef456f56c789554673a7eae6",
    "GSM3442012_E14_M_B7.dge.txt.gz": "9ab6b1eaed93fc0f2011e0b9ec34c07d369aa8a2a0d9012d848f363cd422c988",
    "GSM3442013_E14_M_B8.dge.txt.gz": "55ca1009aed0513269d96550a043877ff2ec01f1a43bd1556e80ed7830793175",
    "GSM3442014_P5_M_A1.dge.txt.gz": "a9cb0cd67cf4f6f05e06562ab636136953a37a2bc10c0deaddd817649c4cdb62",
    "GSM3442015_P5_F_A2.dge.txt.gz": "0e30ddab6dcd9708203db636676fe745cc4abef3304f3fd2df5264a0b10ccd7c",
    "GSM3442016_P4_M_A4.dge.txt.gz": "408158ac92dc98194c9b4b6ddd6cc2f388c672f08958f00c0fbfca0db22704e4",
    "GSM3442017_P4_M_A5.dge.txt.gz": "9405b04a0ba97a1d67b63a66eed96cbeba99574b28ab7bde22de523d979de65f",
    "GSM3442018_P4_F_A6.dge.txt.gz": "ad60a2e4460f5dfda8b63aa7b1f9438ea355948c8ac32023fb953a1fd89a9420",
    "GSM3442019_P4_F_B3.dge.txt.gz": "2d760ae705c95ba3f9292246a2fc3a34c0b3e9029666c395aad653030b3cc709",
    "GSM3442020_P4_F_B4.dge.txt.gz": "c00210c299fbf8c9d8a9c5f532ab992c34a6dbf54b0e453a7db28e34f980ce07",
    "GSM3442021_P4_M_B5.dge.txt.gz": "cc5c89aa37cdeb35f9c66020514713d3a73535ca38b45d650e8288823db6724f",
    "GSM3442022_P30_Male_1.dge.txt.gz": "2ae4016af7d02406db55a2a6439881cab32f6e24c67b0d11bc76d4caecf6833c",
    "GSM3442023_P30_Male_2.dge.txt.gz": "5dd78d65edb04c9e1db68542e1085a68e3d5a3d8d9b9fe4de104320c87556224",
    "GSM3442024_P30_male_3.dge.txt.gz": "a961b7f233c30ad87b8388a0364c6329badeeacb89bf1f2f3b3afa6d5cbb68fd",
    "GSM3442025_P30_male_4.dge.txt.gz": "fd29d40a8c4d8a0c4248efef429208bfc9c31e35d1f789c90f1aa5f5ac2594fc",
    "GSM3442026_P100_Male_1.dge.txt.gz": "61fd005b484088d60be7013ba5b5a170dc1c331a53573bb656eb014c5dc111f3",
    "GSM3442027_P100_Male_2.dge.txt.gz": "a3cbd97d286ec656da885f45753fd1cd1a8ac2a432f147d717097243946cb5c4",
    "GSM3442028_P100_female_1.dge.txt.gz": "4b15341a9ffe955b7de9ec17b81db8e74ac275ed41410f2bda82f635e2605627",
    "GSM3442029_P100_female_2.dge.txt.gz": "a815a00e0c387839bd98166836943a42ecd65f3274c849f21e1fc64dcf909605",
    "GSM3442030_P100_male_3.dge.txt.gz": "bc279a316048245f74aac0d5ce1b1388d5d829f73a44f46b67b32cc006f588e7",
    "GSM3442031_P100_male_4.dge.txt.gz": "8f541309c9695bab208aba3207afb28569f714cddc4367093d233578e616d1a3",
    "GSM3442032_P100_female_3.dge.txt.gz": "99ac502398331efa601aef747ccfa985de5ccdc43c6d98ef77753c02a620c6e4",
    "GSM3442033_P100_female_4.dge.txt.gz": "f1d8a6a17909833c5492da37a224f5e61536d6051baa9080524625258fc11318",
    "GSM3442034_Old_male_1.dge.txt.gz": "299caa74ea5452d320c9e5f957edb0359aa164b935cf57bea0c98112a3dbbcae",
    "GSM3442035_Old_male_2.dge.txt.gz": "ec81b761873253a31ccc1fab09154fbde8295a8f7e28daf2072d7884a445ee36",
    "GSM3442036_Old_male_3.dge.txt.gz": "96caaa56c2d13e7353fc56ae3fb22bd438a96c2f66c2e8c904b8a6eb5cc48de2",
    "GSM3442037_Old_male_4.dge.txt.gz": "88500236871ed276f2c2a614d2777e472232d5fbc7c4f519ddab3e67222a678b",
    "GSM3442038_P100_M_A1.dge.txt.gz": "050067aa6ce1a875d02bc615b9c626ac471050ac7b7b3fcd3947f95a3c2a5127",
    "GSM3442039_P100_M_A2.dge.txt.gz": "6e30cf9f7deba00dcc75dff9198d605d34c69fef52fb594fc0ad36b300bb0d53",
    "GSM3442040_P100_M_B5.dge.txt.gz": "b73408a27faf441b9ef53bea6fefb7efefabef65612775fd9c7073d322ceb000",
    "GSM3442041_P100_M_SALINE_A3.dge.txt.gz": "d52f5f0e625c020c1d07baf2d36ea626cb29d818685833bc70f9add599282f3b",
    "GSM3442042_P100_M_SALINE_A5.dge.txt.gz": "a052b6a2a2e05b9b7af5935c363686d379ce2583fa08668e7c74ffe5016062fd",
    "GSM3442043_P100_M_SALINE_B9.dge.txt.gz": "559a8194411275a1731c1d98bf2333b536a83d5b0d441c0565f7c0a93d87837a",
    "GSM3442044_P100_M_LPC_A4.dge.txt.gz": "9bac3d4db868d900cc4990fea04596a68aa464cfbb6ddb797754ec0326e2e41a",
    "GSM3442045_P100_M_LPC_A6.dge.txt.gz": "db80e71e326daf72a4ca164f855b89d0169a71b2626e08d2af4f7197e2d73780",
    "GSM3442046_P100_M_LPC_B10.dge.txt.gz": "86560a136585b81a9acf24fca293e00825a478a40df3fe0bf73ad0f8cdd8c6c6",
    "GSM3442047_P5_female_nopercoll_1.dge.txt.gz": "8ad73f33b6e0383b0fd571a7778460e9aa619396fb2119f66d61f07835041cc2",
    "GSM3442048_P5_female_nopercoll_2.dge.txt.gz": "4389fb785f4b963a685efabc0fcfcd1cb39ede3cc0d11a70ca6fd24802762809",
    "GSM3442049_P5_female_nopercoll_3.dge.txt.gz": "0fece6118759c729c65e693ad3ba3c42d788e04e760bd71fd296ac0c86a94934",
    "GSM3442050_P5_female_percoll_1.dge.txt.gz": "a42971b13266b6497b6cc5bd0e98c27a651ed1d3323abbbf968e98bc0398ea6a",
    "GSM3442051_P5_female_percoll_2.dge.txt.gz": "f719140563ad48c959400d968ab5902f3bb77949f7d81efe151748d5bd457e90",
    "GSM3442052_P5_female_percoll_3.dge.txt.gz": "24d70830f59bab8c9d6e64b9d6cf6d99dc6711ec02a199088848f677ca7e866a",
}

# Reemst et al. 2022, GEO GSE207067: the series matrix (sample metadata) and one count file per
# sample (Ensembl id, symbol, raw count).
REEMST_SERIES_MATRIX = "GSE207067_series_matrix.txt.gz"
REEMST_SERIES_MATRIX_SHA256 = "0462053a16f17d4d8f2ef8bc8b5424fdbf7b55de6e40698c9f4f80f008157305"
REEMST_FILES = {
    "GSM6276619_104399-001-001.count.txt.gz": "fa98493e9aa74249bbd31d4efc611fe5683cd17668fac867d14f1577b3897493",
    "GSM6276620_104399-001-002.count.txt.gz": "314b742cc5b9bf1e2c1f9f68b4ed93e9afe929cdfcd88550f0c065f07239a29d",
    "GSM6276621_104399-001-003.count.txt.gz": "3bd04d47bf4af07bb6e604f90e96279cc9816b758f158d256660e98e503fd494",
    "GSM6276622_104399-001-004.count.txt.gz": "ecec17f7fd1e5cd1ee73f1913cb13073931b33ef8ee213f68ee6253ff1724c21",
    "GSM6276623_104399-001-005.count.txt.gz": "59b71f90d1ab5adb8a8bd0ced035c43c403deb05d78c5a7caf4e747462d42c19",
    "GSM6276624_104399-001-006.count.txt.gz": "5be346ac08547bd8836eeed258e3ee47afe4f2ae5aec77a474407dc5fc31ed02",
    "GSM6276625_104399-001-007.count.txt.gz": "8dc6529455a75417d024d0117023db50671053ba765eddcefa2c3b82ea181b5e",
    "GSM6276626_104399-001-008.count.txt.gz": "55004e86922e0a312b6abc96ec90950498e833bed9c4e1c3a5594d94e9c9ad58",
    "GSM6276627_104399-001-009.count.txt.gz": "c4208994a088b03c225146c750605fe4b007cd593d49d62d9677f25755336c33",
    "GSM6276628_104399-001-010.count.txt.gz": "be0b0844ad37072b0c1cea65d740179714f135d4b9f9a6c04f8fce56f772e982",
    "GSM6276629_104399-001-011.count.txt.gz": "6f36a42f9e08b6dfb6d7e3828956c8ed4494729dd124235a0185eb56d4e68753",
    "GSM6276630_104399-001-012.count.txt.gz": "be92c839843dd84d48ef7d85a672a6d3694c5cf8098047aa533f89686d7a5262",
    "GSM6276631_104399-001-013.count.txt.gz": "50837558591d26349d52f5a4941a59f49d73a05d257a2a32d06a8f3faeb9fcc5",
    "GSM6276632_104399-001-014.count.txt.gz": "868fa691309ba6fe148557798adc555f24d24fc84a88a9bba7be389902cf48ef",
    "GSM6276633_104399-001-015.count.txt.gz": "9d270f87ec7f4eb354f5ac20b25904e1b85827590231bd168088a93b7770810a",
    "GSM6276634_104399-001-016.count.txt.gz": "6f15d7f28c52a630be0a1fa2e9df0de1a699d4c26b443aa576af94c34a164825",
    "GSM6276635_104399-001-017.count.txt.gz": "03ebab135957fdbc5195f09a4174454971018bb6eb7b60f755487b8836cdd36b",
    "GSM6276636_104399-001-018.count.txt.gz": "0e91f0cc70901494adf167a0f6c328cd6b33b80c56dd61d557be055ff2c451a9",
    "GSM6276637_104399-001-019.count.txt.gz": "59fbe6495cd6c6445897615454503a8fbce7f0a671be68f098e8e1f4e1e87db9",
    "GSM6276638_104399-001-020.count.txt.gz": "5b5a68004392b06a774e13d7fd90c1c841aae636929b801eaa2a442dc97be4ed",
    "GSM6276639_104399-001-021.count.txt.gz": "da51a4bbf006f8697bf434da02471c27a7fc436fbb01c56d7b63df53866b7f57",
    "GSM6276640_104399-001-022.count.txt.gz": "3fb333b73c8a8c2e3f6e3375344ccaf311f4e5819ff07f8dcc0e53da3430d890",
    "GSM6276641_104399-001-023.count.txt.gz": "b0974f357c3e3982a83ab6945cb9842eb40989e2b27310943b6c1c06810b0294",
    "GSM6276642_104399-001-024.count.txt.gz": "79e11503ffd25dded01ee79308f0120cb934e893338666e8efbd584df8fe6a8c",
    "GSM6276643_104399-001-025.count.txt.gz": "e3785fa85d34527f54fec0dfec5bad980f40c02a46d45afb93a3bfbc60005dde",
    "GSM6276644_104399-001-026.count.txt.gz": "40e81f03cf605d23f7bea2f7fe6bd378e04f62a323610b3bfa157861bd6b3620",
    "GSM6276645_104399-001-027.count.txt.gz": "b0f2bdffc8d0ccd91b1cf2608ea8dd219db175bf5f45d7bcf4aa454a16b390af",
    "GSM6276646_104399-001-028.count.txt.gz": "7834d6527dd457a3c0ef0043fe98f4480667859227d8b52dd8dcc1de960a71fe",
    "GSM6276647_104399-001-029.count.txt.gz": "a4137b5633d3596fc5e33cc5452462981748a89d065d8605ca2495c058aba6fa",
    "GSM6276648_104399-001-030.count.txt.gz": "e92da499508533f0216891b144cfe43236c4d1e63e2b721f0c3f64b644ee1517",
    "GSM6276649_104399-001-031.count.txt.gz": "c553c44f5b6063705a3d2be9d5b7ca4bb336326f89525497ea424058b4a8f94a",
    "GSM6276650_104399-001-032.count.txt.gz": "eaf1f9484c38efcdf35a2f12fa7ab0081b64400acc00f1dd9ed260764128df24",
    "GSM6276651_104399-001-033.count.txt.gz": "d24615f737f4eec4e0d74042855281162b033ef028637c703eac50ca4e00da96",
    "GSM6276652_104399-001-034.count.txt.gz": "dfdaf1f141d945f81a22e9a44262b4cd612d5bd46fedd9c955082ccc02c9923e",
    "GSM6276653_104399-001-035.count.txt.gz": "e04ebd5ff1de62e72ad2cc10c8ebf46f00c1ae588b3691f095023c89f4d416d2",
    "GSM6276654_104399-001-036.count.txt.gz": "11801a9256f2e41a23ac415692bf8686db88ffbf12a9a9643aa71bf8766daf95",
    "GSM6276655_104399-001-037.count.txt.gz": "b6ed2cfdb63575ece9f076bf33d9c0eda92145fd086c61f148bade9e5fc2920d",
    "GSM6276656_104399-001-038.count.txt.gz": "6303a8da3021baa26af78478763d07792433cc9ddee8be95f39749f133faeb9f",
    "GSM6276657_104399-001-039.count.txt.gz": "dfbf71e9b7417cf9b56a5117ca8ca0f2983428c6973518be63c1650340220be0",
    "GSM6276658_104399-001-040.count.txt.gz": "21a915cfb26b01928f0f3ea6f428e8279a2b0108702c82a7180aea0cdd80ac48",
    "GSM6276659_104399-001-041.count.txt.gz": "3428e58f4b2a762e90e53342ef8c9a1629c64351d3b28bc91d16d2dbcf16bf00",
    "GSM6276660_104399-001-043.count.txt.gz": "9c678eac53fd37c3dd0e065418745e1ae48d160c232117f6d057f2e2d4bdff8f",
    "GSM6276661_104399-001-044.count.txt.gz": "3034b53312bf2ab9ea08835c80d8b07fc2aefc570295a5f2681e6717986e6213",
    "GSM6276662_104399-001-045.count.txt.gz": "a61910504c589f2b64165434b8aaacad14b99cb81d58f6ffcc884b138740168f",
    "GSM6276663_104399-001-046.count.txt.gz": "49b204664cac008ced0a6588b7a54f08dbc30ae4733b3d51fbdfd53e458775cf",
    "GSM6276664_104399-001-047.count.txt.gz": "711ec0190b2a06fc751eee1fafc98469963c26300242afd627e9ad8184f8f93c",
    "GSM6276665_104399-001-048.count.txt.gz": "a3815949eeae5cc90f07f7de6be67379ee593d074a52d1b67651ab58e671a932",
    "GSM6276666_104399-001-049.count.txt.gz": "ba856ba8e53f4976f9c4da3f0c219d051ffed4c79f7bc22a12ee6607f6fe4e33",
    "GSM6276667_104399-001-050.count.txt.gz": "28854bdd6209beb6f4fd610d820aa08de552dae0418df8951e5269b3c5aa0f58",
    "GSM6276668_104399-001-051.count.txt.gz": "aa8b8bad38be9a7a2dc90e262f9530016d08c6a41621ae12e098c162676b701a",
}

# Schaum et al. 2020 (Tabula Muris Senis bulk), GEO GSE132040.
TMS_BULK_COUNTS = "GSE132040_190214_A00111_0269_AHH3J3DSXX_190214_A00111_0270_BHHMFWDSXX.csv.gz"
TMS_BULK_METADATA = "GSE132040_MACA_Bulk_metadata.csv.gz"

# Tabula Muris Senis 10x droplet, CZ CELLxGENE collection 0b9d8a04-bb9d-44da-aa27-705bb65b54eb.
# The file name in each URL is the CELLxGENE dataset_version_id, which pins the release.
TMS_SINGLE_CELL = {
    "Marrow": ("https://datasets.cellxgene.cziscience.com/88dcbe63-d6eb-4fae-9362-e21b98365a42.h5ad",
               "f87288dda2442fa7815b6f9971ab0bec73c767d0a377a977864da11bc49a46a5"),
    "Lung": ("https://datasets.cellxgene.cziscience.com/e67a08e2-96fc-4244-b8a3-fca092b22f77.h5ad",
             "406374e0dce32f10d7c3757f2b1c81fc5ae307b97db8ebdb9787938eef997351"),
    "Kidney": ("https://datasets.cellxgene.cziscience.com/cfd31d35-ceb2-4990-b0b5-3434ecbb7967.h5ad",
               "02d37b0cc51c2aaaf2cc27c3a2ddadd4d97bafebe247c37d64a363226ef5c6b4"),
    "Limb_Muscle": ("https://datasets.cellxgene.cziscience.com/a0560e6a-33db-4b82-8666-ba5480b4e23f.h5ad",
                    "3c9d093eb390c7bdc565e114f3bb24ec1e64d8a938bf900e8d8bdaf2a08f4355"),
}

# NYGC ALS Consortium + Target ALS motor cortex bulk (Prudencio et al. 2020), GEO GSE153960.
NYGC_COUNTS = "GSE153960_Gene_counts_matrix_RSEM_Prudencio_et_al_2020.txt.gz"
NYGC_SERIES_MATRICES = ("GSE153960-GPL16791_series_matrix.txt.gz", "GSE153960-GPL24676_series_matrix.txt.gz")

# Pineda et al. 2024 single-nucleus data, Synapse project syn51105515 (open; a free Synapse
# account and a personal access token in ~/.synapseConfig are needed to download). The
# sample table (S01) and the 49 motor-cortex (BA4) libraries of ALS, C9ALS and pathologically
# normal donors, each a folder of counts_fil.mtx, row_metadata.tsv and col_metadata.tsv.
# Every file is pinned by its Synapse id and version, Synapse's MD5, and our SHA-256
# (pineda_synapse_files.tsv; library = the folder).
PINEDA_SAMPLE_TABLE = ("S01_sample_information.xlsx", "syn53497707", 1,
                       "1b35d9e34628f6ba87c3c4360e9f401079bb46290ba1d9d4c03d46283243dcd6")
PINEDA_FILES_TSV = Path(__file__).resolve().parent / "pineda_synapse_files.tsv"


def pineda_files() -> pd.DataFrame:
    return pd.read_csv(PINEDA_FILES_TSV, sep="\t", dtype=str, keep_default_na=False)


# Ensembl release 111: mouse -> human one-to-one orthologs (Compara protein_default
# homologies) and human gene names, for the foundation models on the mouse pairs. Geneformer
# documents no Ensembl release; 111 is the release whose human gene set best covers its
# gc104M dictionary (231 of 20,271 ids missing; 233-318 for releases 105-116). Ensembl stores
# each mouse-human pair in only one of the two species' files, so both are needed.
ENSEMBL_RELEASE = 111
ENSEMBL = f"https://ftp.ensembl.org/pub/release-{ENSEMBL_RELEASE}"
ORTHOLOG_FILES = {
    "mouse": f"Compara.{ENSEMBL_RELEASE}.protein_default.homologies.mus_musculus.tsv.gz",
    "human": f"Compara.{ENSEMBL_RELEASE}.protein_default.homologies.homo_sapiens.tsv.gz",
}
HUMAN_GTF = f"Homo_sapiens.GRCh38.{ENSEMBL_RELEASE}.gtf.gz"

# Human blood, lupus: Perez et al. 2022 PBMC (CELLxGENE dataset version c55dc602) and Hung
# et al. 2015 whole blood via recount3 (SRP062966; G026 annotation). Human ileum, Crohn's:
# IBDverse (CELLxGENE dataset version e3460364) and MSCCR (GEO GSE193677), gene names from
# Ensembl 75 (the counts' GRCh37.75 ids). CELLxGENE and recount3 publish no SHA-256; ours
# were computed at download (2026-10-05).
CELLXGENE = "https://datasets.cellxgene.cziscience.com/{}.h5ad"
RECOUNT3 = "https://recount-opendata.s3.amazonaws.com/recount3/release/human"
LUPUS_SINGLE_CELL = "perez2022_c55dc602-d168-4d15-acc1-5de4f2f5d551.h5ad"
LUPUS_BULK_COUNTS = "sra.gene_sums.SRP062966.G026.gz"
LUPUS_BULK_METADATA = "sra.sra.SRP062966.MD.gz"
LUPUS_BULK_GENES = "human.gene_sums.G026.gtf.gz"
CROHNS_SINGLE_CELL = "ibdverse_e3460364-c300-4616-9923-3a0224894998.h5ad"
CROHNS_BULK_COUNTS = "GSE193677_MSCCR_Biopsy_counts.txt.gz"
CROHNS_BULK_METADATA = "GSE193677_series_matrix.txt.gz"
CROHNS_BULK_GENES = "Homo_sapiens.GRCh37.75.gtf.gz"

# Each file is stored under <outputs>/01_download/<directory>/<file name>. A pair's data
# directories (its stage-02 inputs), and the ortholog directory its stage 04 (panel) also reads.
PAIR_DIRECTORIES = {"hammond": ("hammond", "reemst"), "soundlife": ("soundlife",), "tms": ("tms",),
                    "pineda": ("pineda", "nygc"), "lupus": ("lupus",), "crohns": ("crohns",)}
ORTHOLOG_DIRECTORIES = {"hammond": ("ensembl",), "tms": ("ensembl",), "soundlife": (), "pineda": (), "lupus": (), "crohns": ()}

# (directory, file name) -> (url, sha256) for every scripted download.
DOWNLOADS = {
    **{("hammond", name): (geo_sample_url(name), sha) for name, sha in HAMMOND_FILES.items()},
    ("reemst", REEMST_SERIES_MATRIX): (geo_series_url("GSE207067", "matrix", REEMST_SERIES_MATRIX),
                                       REEMST_SERIES_MATRIX_SHA256),
    **{("reemst", name): (geo_sample_url(name), sha) for name, sha in REEMST_FILES.items()},
    **{("tms", f"tms_droplet_{organ.lower()}.h5ad"): spec for organ, spec in TMS_SINGLE_CELL.items()},
    ("tms", TMS_BULK_COUNTS): (geo_series_url("GSE132040", "suppl", TMS_BULK_COUNTS),
                               "f42e5df6722093f85bd3bd6c2d09823a24a2cbd94894452edffb55f2e159fc65"),
    ("tms", TMS_BULK_METADATA): (geo_series_url("GSE132040", "suppl", TMS_BULK_METADATA),
                                 "d2cb023abc8399401e227da38d040a15ab4395cee10dca36a812fa59399b4430"),
    ("nygc", NYGC_COUNTS): (geo_series_url("GSE153960", "suppl", NYGC_COUNTS),
                              "458e332c859832f586676a0502357191b6dc39404ae298ab0299f3420282b805"),
    ("nygc", NYGC_SERIES_MATRICES[0]): (geo_series_url("GSE153960", "matrix", NYGC_SERIES_MATRICES[0]),
                                          "f850e6e939915b542cd03596505a5fb3952ab2d30b4dccadea92a905f4e2aed1"),
    ("nygc", NYGC_SERIES_MATRICES[1]): (geo_series_url("GSE153960", "matrix", NYGC_SERIES_MATRICES[1]),
                                          "c6ac4856cd9ad12d055ef296b685959cd4f5464f29c6f2e7a0825fbe9af751c9"),
    # saved under a name with the species, since Ensembl names both files the same
    ("ensembl", ORTHOLOG_FILES["mouse"]): (
        f"{ENSEMBL}/tsv/ensembl-compara/homologies/mus_musculus/Compara.{ENSEMBL_RELEASE}.protein_default.homologies.tsv.gz",
        "5e7308add5f008d96d396787baa101c87034c7143b0d8264e686774690fb5b56"),
    ("ensembl", ORTHOLOG_FILES["human"]): (
        f"{ENSEMBL}/tsv/ensembl-compara/homologies/homo_sapiens/Compara.{ENSEMBL_RELEASE}.protein_default.homologies.tsv.gz",
        "0f27e156fab29d93ee7c2981a758cc13ca7bfc6f7065f891252c50fd720d9caf"),
    ("ensembl", HUMAN_GTF): (f"{ENSEMBL}/gtf/homo_sapiens/{HUMAN_GTF}",
                             "a52356765a41264e17a2076aff1abab703ec3ba239b78f88560756e85c169831"),
    ("lupus", LUPUS_SINGLE_CELL): (CELLXGENE.format("c55dc602-d168-4d15-acc1-5de4f2f5d551"),
                                   "3c0b74d54c03838a49817edce95314e6a4fa048ef35b74d1e697b8b2fd07cc03"),
    ("lupus", LUPUS_BULK_COUNTS): (f"{RECOUNT3}/data_sources/sra/gene_sums/66/SRP062966/{LUPUS_BULK_COUNTS}",
                                   "ce7418b6af815013946f4206555eb156d434fe9a1898769cdbc24026ca67658b"),
    ("lupus", LUPUS_BULK_METADATA): (f"{RECOUNT3}/data_sources/sra/metadata/66/SRP062966/{LUPUS_BULK_METADATA}",
                                     "22db2c2e5883bc256344de2eca3d10a4b3bb0738f48aea30e5dbedeac6c402c9"),
    ("lupus", LUPUS_BULK_GENES): (f"{RECOUNT3}/annotations/gene_sums/{LUPUS_BULK_GENES}",
                                  "2d1bdff8d91e08f1291fd47850d9d61cfa6768cc3a4cb8516775a873b007d0f6"),
    ("crohns", CROHNS_SINGLE_CELL): (CELLXGENE.format("e3460364-c300-4616-9923-3a0224894998"),
                                     "e93b382b74a53205813afaf875189052051582fde47921bd07f8e6fb3a2ab71c"),
    ("crohns", CROHNS_BULK_COUNTS): (geo_series_url("GSE193677", "suppl", CROHNS_BULK_COUNTS),
                                     "94c85e2f97b5151292cf52f0792011521f11c456b9d4d9d3cdb9722683a8ced3"),
    ("crohns", CROHNS_BULK_METADATA): (geo_series_url("GSE193677", "matrix", CROHNS_BULK_METADATA),
                                       "7b155e6b1c7d1c39343c06f39b1ca4cff8af2d6ea7e92ed2e057f42c49bac362"),
    ("crohns", CROHNS_BULK_GENES): ("https://ftp.ensembl.org/pub/release-75/gtf/homo_sapiens/" + CROHNS_BULK_GENES,
                                    "770f48d9aecd9bdecdf147fd141c23670b215bccbdbfa459048f45b02581c2ed"),
}

# AIFI Sound Life (Gong et al. 2025; the bulk also Smithmyer et al. 2025; data doi:10.57785/bn17-3h16; no
# license is stated). The portal serves files to a browser only.
SOUNDLIFE_PORTAL = "https://apps.allenimmunology.org/aifi/insights/dynamics-imm-health-age/downloads/"
SOUNDLIFE_SINGLE_CELL = tuple(f"SoundLife_{age}Adult_{sex}_CMV{cmv}.h5ad"
                              for age in ("Young", "Older") for sex in ("Male", "Female") for cmv in ("pos", "neg"))
SOUNDLIFE_BULK = "sound-life_whole-blood_no-stim.h5ad"

# The single-cell input is one slim file per portal file (a random 3,000 cells per sample;
# see sc2bulk.data.soundlife and pipeline/00_soundlife_slims.py, which rebuilds them from the
# portal files). File checksums verify our copies; content checksums verify a rebuild.
SOUNDLIFE_SLIM_FILES = {
    "SoundLife_YoungAdult_Male_CMVpos__slim.h5ad": "c69c7c0c3627960d72ba1c045aa9657fe77985ae9ca8c004c2a1d502046cb403",
    "SoundLife_YoungAdult_Male_CMVneg__slim.h5ad": "4388659968a9443a9f5b14d1d7d7cb4df37393642c20ab05e70b988e2439676f",
    "SoundLife_YoungAdult_Female_CMVpos__slim.h5ad": "5d50358a4ccaaaf800b0770fb445561e0dc226fe39be113b00937c1a2cae7def",
    "SoundLife_YoungAdult_Female_CMVneg__slim.h5ad": "1679271602877b2aa6f96e198e119e6dbeceabcf651261b0c915bef779dace30",
    "SoundLife_OlderAdult_Male_CMVpos__slim.h5ad": "11d0d192bbb7c10af2c97ded674da6d7cf051cd3c135b50a09944243025cf27f",
    "SoundLife_OlderAdult_Male_CMVneg__slim.h5ad": "f273d2b0fee9255f5e0567f1a7d3ecb87369dc3d3b017ed35313cf0d749b935a",
    "SoundLife_OlderAdult_Female_CMVpos__slim.h5ad": "f273d45b4a244b3070148a56fe2dd26a840f1863809c78904b5c01311e869b98",
    "SoundLife_OlderAdult_Female_CMVneg__slim.h5ad": "adb1be7cde3f01a8449aa5895a9f6c6e67c8111956be8894e95f9693bbad19e1",
}
SOUNDLIFE_SLIM_CONTENT = {
    "SoundLife_YoungAdult_Male_CMVpos__slim.h5ad": "52717e0628c4e6432dcfbcf39ad946c212de6d8935a6078bd9dbcbe14ba2b400",
    "SoundLife_YoungAdult_Male_CMVneg__slim.h5ad": "ef36ee767bf4aba07fea7956038659b0d2d1506b9db094a4b0c89be266c18ddf",
    "SoundLife_YoungAdult_Female_CMVpos__slim.h5ad": "be9d8a921f3fb6fd8bda2794e0f9734fce8dca5a3d74c91ded2ab8f0f5396f0a",
    "SoundLife_YoungAdult_Female_CMVneg__slim.h5ad": "1461acffd25cd7aacce1e97333e066077b79f4081e434e0bc0ffe42f8f71f215",
    "SoundLife_OlderAdult_Male_CMVpos__slim.h5ad": "31c3810241836a81fb3347cb101127bfe91cc9d7e1a307dff3bf775164a748eb",
    "SoundLife_OlderAdult_Male_CMVneg__slim.h5ad": "f285d69694b6b1b429dd2b03f1a511b22f02ee6dd77e1f057ad3e1666bf55915",
    "SoundLife_OlderAdult_Female_CMVpos__slim.h5ad": "26829620626600f56f5a03933558e08c4a2483dda6f3bb33d2006bb3ece4ebd8",
    "SoundLife_OlderAdult_Female_CMVneg__slim.h5ad": "2e1f1902bf70cdda37407480eb8d2e51db143618319c3a5f6e85d2d25cc386cb",
}

# (directory, file name) -> (synapse id, version, sha256) for every Synapse download; the file
# name of a library file is "<library>/<file>".
SYNAPSE = {
    ("pineda", PINEDA_SAMPLE_TABLE[0]): PINEDA_SAMPLE_TABLE[1:],
    **{("pineda", f"{r.library}/{r.file}"): (r.synapse_id, int(r.version), r.sha256) for r in pineda_files().itertuples()},
}

# (directory, file name) -> (instructions, sha256).
MANUAL = {
    **{("soundlife", name): (f"Rebuild it with pipeline/00_soundlife_slims.py from the portal files at {SOUNDLIFE_PORTAL} "
                             "(scRNA-seq, Sound Life subject groups)", sha)
       for name, sha in SOUNDLIFE_SLIM_FILES.items()},
    ("soundlife", SOUNDLIFE_BULK): (f"Download '{SOUNDLIFE_BULK}' (bulk RNA-seq) from {SOUNDLIFE_PORTAL}",
                                    "980d39ca002d1fd18c0e7cb6d658602d8395e1cf7fe8a2fac069b12bfb7d71e9"),
}

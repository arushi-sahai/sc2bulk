"""The set encoders' networks (sc2bulk.harness trains them): each module's `network(config, in_dim)`
returns an nn.Module with embed(x, weights) -> (bags, d) and head(embedding) -> (bags,) logits."""

# Fork provenance

This fork starts at upstream TensorFold `b472b9f84` (release 1.3.2, 2026-11-20) and adds
27 patches as individual commits. Each commit keeps its author; the fork's own commits are
authored by Jens Lewald. The served build is proven end to end against GLM-5.3-Flash on two
GB10s (TP=2): the full quality battery, a resume probe at 73,012 tokens and a tool-call-guard
probe all pass, with zero crashes.

| Commit | Author | What |
|---|---|---|
| `native: beside GLM's host tier, a device prompt cache` | MiaAI-Lab | device-side prompt cache beside GLM's host tier |
| `glm cuda: TF_GLM_W8A8=1 (or 32), packed 8-bit prompt matmuls` | MiaAI-Lab | packed 8-bit prompt matmuls |
| `tools: share_gate.py, the served exactness gate` | GrgaPalcic | the served-exactness gate tool |
| `qwen cuda vision: ModelOpt MXFP8 and NVFP4 tower linears` | BadAd84 | quantized vision-tower linears load |
| `glm cuda: sub-grid prompt-cache marks at the page edge under a short prompt` | Jens Lewald | sub-grid cache marks for GLM |
| `glm cuda: the page edge serves only a prompt; the pass carries on its own cuts` | Jens Lewald | the page edge's serving rule |
| `0093 port: debug-log the PastCapacity throw` | Jens Lewald | port of the PastCapacity diagnostics |
| `0094 port: the request's page grant carries slack` | Jens Lewald | the grant-slack fix (the PastCapacity wall) |
| `kv-shard graft, gate 1 … gate 5d` (15 commits) | Jens Lewald | the KV shard, grafted gate by gate — each gate compiled and proven on its own |
| `the tool-call guard on 1.3.2: the fork's client protection` | Jens Lewald | `TF_TOOL_CALL_GUARD=1`: the closing note for tool-less replies over tool history, and the drop of well-formed calls to unoffered tools |
| `the dflash selector skips out-of-vocabulary candidate ids` | Jens Lewald | the no-winner guard (`0x7fffffff`, NaN or -inf logits): never index the codebooks with it, and clamp a corrupt anchor — ported from the MiaAI recipe's selector fix; the fork's shard design already excludes the corruption sources upstream hit, so this is defense in depth (pending the next image rebuild) |

Carried upstream pull requests keep their authors' names (MiaAI-Lab, GrgaPalcic, BadAd84);
their licence terms remain in [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md) and the
licence list under [LICENSES/](../LICENSES/). The engine itself is TensorFold by Ash Hart and
contributors, Apache License 2.0 — this fork changes nothing about that licence.

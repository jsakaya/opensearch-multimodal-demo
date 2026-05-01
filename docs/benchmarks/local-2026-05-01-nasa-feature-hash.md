# OpenLens Retrieval Benchmark

- Generated: `2026-05-01T19:15:57Z`
- Index: `openlens_multimodal`
- OpenSearch: `OpenSearch 3.6.0 at http://localhost:9200`
- Docs: `10000`
- Embeddings: `feature-hash` `384d`

| Slice | n | p50 ms | p95 ms | exact@k | modality@1 | modality@3 |
|---|---:|---:|---:|---:|---:|---:|
| all | 366 | 41.12 | 197.52 | 96.00% | 87.70% | 95.90% |
| hybrid | 90 | 97.20 | 223.66 | 96.00% | 90.00% | 96.67% |
| keyword | 90 | 13.09 | 31.35 | 92.00% | 90.00% | 96.67% |
| lir | 90 | 72.19 | 199.67 | 96.00% | 83.33% | 93.33% |
| sql | 6 | 16.65 | 318.17 | 0.00% | 100.00% | 100.00% |
| vector | 90 | 8.68 | 16.21 | 100.00% | 86.67% | 96.67% |

## By Expected Modality

| Modality | n | p50 ms | p95 ms | exact@k | modality@1 | modality@3 |
|---|---:|---:|---:|---:|---:|---:|
| audio | 72 | 57.38 | 227.29 | 95.00% | 91.67% | 100.00% |
| image | 72 | 16.02 | 102.74 | 85.00% | 91.67% | 100.00% |
| pdf | 72 | 16.58 | 97.01 | 100.00% | 95.83% | 95.83% |
| table | 78 | 20.61 | 200.29 | 100.00% | 88.46% | 88.46% |
| video | 72 | 15.71 | 96.18 | 100.00% | 70.83% | 95.83% |

## By Mode And Expected Modality

| Modality | Mode | n | p50 ms | p95 ms | exact@k | modality@1 | modality@3 |
|---|---|---:|---:|---:|---:|---:|---:|
| audio | hybrid | 18 | 212.18 | 265.62 | 100.00% | 100.00% | 100.00% |
| audio | keyword | 18 | 28.38 | 57.38 | 80.00% | 66.67% | 100.00% |
| audio | lir | 18 | 123.01 | 137.88 | 100.00% | 100.00% | 100.00% |
| audio | vector | 18 | 15.22 | 17.61 | 100.00% | 100.00% | 100.00% |
| image | hybrid | 18 | 95.38 | 120.96 | 80.00% | 100.00% | 100.00% |
| image | keyword | 18 | 12.42 | 16.02 | 80.00% | 100.00% | 100.00% |
| image | lir | 18 | 71.39 | 79.59 | 80.00% | 83.33% | 100.00% |
| image | vector | 18 | 7.78 | 10.38 | 100.00% | 83.33% | 100.00% |
| pdf | hybrid | 18 | 92.05 | 134.79 | 100.00% | 100.00% | 100.00% |
| pdf | keyword | 18 | 11.65 | 16.58 | 100.00% | 100.00% | 100.00% |
| pdf | lir | 18 | 67.19 | 79.92 | 100.00% | 83.33% | 83.33% |
| pdf | vector | 18 | 8.26 | 10.32 | 100.00% | 100.00% | 100.00% |
| table | hybrid | 18 | 180.88 | 186.91 | 100.00% | 83.33% | 83.33% |
| table | keyword | 18 | 16.45 | 20.61 | 100.00% | 83.33% | 83.33% |
| table | lir | 18 | 195.07 | 203.87 | 100.00% | 100.00% | 100.00% |
| table | sql | 6 | 16.65 | 318.17 | 0.00% | 100.00% | 100.00% |
| table | vector | 18 | 13.53 | 19.35 | 100.00% | 83.33% | 83.33% |
| video | hybrid | 18 | 91.12 | 106.57 | 100.00% | 66.67% | 100.00% |
| video | keyword | 18 | 11.56 | 15.71 | 100.00% | 100.00% | 100.00% |
| video | lir | 18 | 63.01 | 72.37 | 100.00% | 50.00% | 83.33% |
| video | vector | 18 | 7.38 | 9.51 | 100.00% | 66.67% | 100.00% |

## Notes

- `exact@k` uses title/metadata queries generated from indexed records.
- `modality@1` and `modality@3` measure whether scenario queries retrieve the expected modality.
- SQL rows use the OpenSearch SQL plugin path.

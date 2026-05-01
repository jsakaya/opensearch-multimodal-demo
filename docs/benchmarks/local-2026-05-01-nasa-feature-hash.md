# OpenLens Retrieval Benchmark

- Generated: `2026-05-01T18:15:28Z`
- Index: `openlens_multimodal`
- OpenSearch: `OpenSearch 3.3.0 at http://localhost:9200`
- Docs: `10000`
- Embeddings: `feature-hash` `384d`

| Slice | n | p50 ms | p95 ms | exact@k | modality@1 | modality@3 |
|---|---:|---:|---:|---:|---:|---:|
| all | 366 | 25.66 | 107.10 | 93.00% | 86.07% | 94.26% |
| hybrid | 90 | 68.92 | 178.98 | 96.00% | 90.00% | 96.67% |
| keyword | 90 | 8.31 | 25.52 | 92.00% | 90.00% | 96.67% |
| lir | 90 | 42.41 | 107.09 | 92.00% | 83.33% | 93.33% |
| sql | 6 | 9.26 | 15.87 | 0.00% | 100.00% | 100.00% |
| vector | 90 | 5.21 | 9.43 | 92.00% | 80.00% | 90.00% |

## By Expected Modality

| Modality | n | p50 ms | p95 ms | exact@k | modality@1 | modality@3 |
|---|---:|---:|---:|---:|---:|---:|
| audio | 72 | 29.48 | 179.06 | 95.00% | 91.67% | 100.00% |
| image | 72 | 10.61 | 73.92 | 85.00% | 91.67% | 100.00% |
| pdf | 72 | 9.31 | 68.38 | 90.00% | 87.50% | 91.67% |
| table | 78 | 12.44 | 107.10 | 100.00% | 88.46% | 88.46% |
| video | 72 | 11.03 | 69.32 | 95.00% | 70.83% | 91.67% |

## By Mode And Expected Modality

| Modality | Mode | n | p50 ms | p95 ms | exact@k | modality@1 | modality@3 |
|---|---|---:|---:|---:|---:|---:|---:|
| audio | hybrid | 18 | 167.47 | 211.68 | 100.00% | 100.00% | 100.00% |
| audio | keyword | 18 | 21.25 | 29.48 | 80.00% | 66.67% | 100.00% |
| audio | lir | 18 | 72.36 | 79.82 | 100.00% | 100.00% | 100.00% |
| audio | vector | 18 | 8.80 | 9.90 | 100.00% | 100.00% | 100.00% |
| image | hybrid | 18 | 68.53 | 91.19 | 80.00% | 100.00% | 100.00% |
| image | keyword | 18 | 7.82 | 10.61 | 80.00% | 100.00% | 100.00% |
| image | lir | 18 | 41.57 | 45.20 | 80.00% | 83.33% | 100.00% |
| image | vector | 18 | 4.49 | 6.24 | 100.00% | 83.33% | 100.00% |
| pdf | hybrid | 18 | 62.29 | 78.76 | 100.00% | 83.33% | 100.00% |
| pdf | keyword | 18 | 7.02 | 9.31 | 100.00% | 100.00% | 100.00% |
| pdf | lir | 18 | 39.01 | 45.62 | 80.00% | 83.33% | 83.33% |
| pdf | vector | 18 | 4.77 | 6.74 | 80.00% | 83.33% | 83.33% |
| table | hybrid | 18 | 101.58 | 105.60 | 100.00% | 83.33% | 83.33% |
| table | keyword | 18 | 9.30 | 12.44 | 100.00% | 83.33% | 83.33% |
| table | lir | 18 | 102.17 | 108.21 | 100.00% | 100.00% | 100.00% |
| table | sql | 6 | 9.26 | 15.87 | 0.00% | 100.00% | 100.00% |
| table | vector | 18 | 8.09 | 11.20 | 100.00% | 83.33% | 83.33% |
| video | hybrid | 18 | 64.62 | 74.42 | 100.00% | 83.33% | 100.00% |
| video | keyword | 18 | 7.22 | 11.03 | 100.00% | 100.00% | 100.00% |
| video | lir | 18 | 37.20 | 42.48 | 100.00% | 50.00% | 83.33% |
| video | vector | 18 | 4.57 | 7.28 | 80.00% | 50.00% | 83.33% |

## Notes

- `exact@k` uses title/metadata queries generated from indexed records.
- `modality@1` and `modality@3` measure whether scenario queries retrieve the expected modality.
- SQL rows use the OpenSearch SQL plugin path.

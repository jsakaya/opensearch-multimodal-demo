# OpenLens Retrieval Benchmark

- Generated: `2026-05-01T17:32:40Z`
- Index: `openlens_multimodal`
- OpenSearch: `OpenSearch 3.3.0 at http://localhost:9200`
- Docs: `10000`
- Embeddings: `feature-hash` `384d`

| Slice | n | p50 ms | p95 ms | exact@k | modality@1 | modality@3 |
|---|---:|---:|---:|---:|---:|---:|
| all | 366 | 18.33 | 107.07 | 84.00% | 90.16% | 95.08% |
| hybrid | 90 | 69.03 | 107.22 | 96.00% | 90.00% | 96.67% |
| keyword | 90 | 7.83 | 17.87 | 96.00% | 86.67% | 90.00% |
| lir | 90 | 43.64 | 109.88 | 68.00% | 86.67% | 96.67% |
| sql | 6 | 3.92 | 15.96 | 0.00% | 100.00% | 100.00% |
| vector | 90 | 5.06 | 9.15 | 76.00% | 96.67% | 96.67% |

## By Expected Modality

| Modality | n | p50 ms | p95 ms | exact@k | modality@1 | modality@3 |
|---|---:|---:|---:|---:|---:|---:|
| audio | 72 | 10.89 | 83.83 | 85.00% | 91.67% | 95.83% |
| image | 72 | 25.65 | 93.17 | 70.00% | 95.83% | 100.00% |
| pdf | 72 | 7.19 | 59.44 | 85.00% | 87.50% | 91.67% |
| table | 78 | 10.37 | 110.07 | 100.00% | 88.46% | 88.46% |
| video | 72 | 14.06 | 71.15 | 80.00% | 87.50% | 100.00% |

## By Mode And Expected Modality

| Modality | Mode | n | p50 ms | p95 ms | exact@k | modality@1 | modality@3 |
|---|---|---:|---:|---:|---:|---:|---:|
| audio | hybrid | 18 | 75.89 | 102.39 | 100.00% | 83.33% | 100.00% |
| audio | keyword | 18 | 8.79 | 10.89 | 100.00% | 83.33% | 83.33% |
| audio | lir | 18 | 54.34 | 59.36 | 60.00% | 100.00% | 100.00% |
| audio | vector | 18 | 5.51 | 7.06 | 80.00% | 100.00% | 100.00% |
| image | hybrid | 18 | 65.60 | 237.23 | 80.00% | 100.00% | 100.00% |
| image | keyword | 18 | 8.59 | 25.65 | 80.00% | 83.33% | 100.00% |
| image | lir | 18 | 42.57 | 47.39 | 60.00% | 100.00% | 100.00% |
| image | vector | 18 | 5.09 | 7.93 | 60.00% | 100.00% | 100.00% |
| pdf | hybrid | 18 | 47.38 | 70.37 | 100.00% | 83.33% | 100.00% |
| pdf | keyword | 18 | 5.94 | 7.19 | 100.00% | 83.33% | 83.33% |
| pdf | lir | 18 | 32.37 | 34.86 | 60.00% | 83.33% | 83.33% |
| pdf | vector | 18 | 3.95 | 4.72 | 80.00% | 100.00% | 100.00% |
| table | hybrid | 18 | 105.75 | 110.69 | 100.00% | 83.33% | 83.33% |
| table | keyword | 18 | 9.44 | 10.30 | 100.00% | 83.33% | 83.33% |
| table | lir | 18 | 108.19 | 110.87 | 100.00% | 100.00% | 100.00% |
| table | sql | 6 | 3.92 | 15.96 | 0.00% | 100.00% | 100.00% |
| table | vector | 18 | 8.10 | 10.37 | 100.00% | 83.33% | 83.33% |
| video | hybrid | 18 | 58.59 | 78.27 | 100.00% | 100.00% | 100.00% |
| video | keyword | 18 | 6.15 | 14.06 | 100.00% | 100.00% | 100.00% |
| video | lir | 18 | 36.49 | 58.75 | 60.00% | 50.00% | 100.00% |
| video | vector | 18 | 4.65 | 10.45 | 60.00% | 100.00% | 100.00% |

## Notes

- `exact@k` uses title/metadata queries generated from indexed records.
- `modality@1` and `modality@3` measure whether scenario queries retrieve the expected modality.
- SQL rows use the OpenSearch SQL plugin path.

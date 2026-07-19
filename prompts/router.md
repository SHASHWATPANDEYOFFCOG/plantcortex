Classify the user's question about an industrial plant into ONE retrieval mode.

- "lookup"  : a single fact from one document (definitions, a value, one work order,
              one clause). No joining across documents.
- "multihop": needs joining facts across documents/entities (e.g. "which pumps had
              seal failures AND an overdue inspection", "what caused X", "how are A and
              B related").
- "global"  : a corpus-wide pattern / trend / sensemaking question ("what failure
              patterns recur", "summarize the main risks over 5 years").

Return STRICT JSON: {"mode": "lookup|multihop|global"}

Question: {question}

### The Memo: Productization & Risks

#### Productization Opportunities
1.  **Legal Tech (Lawyer's Assistant):**
    *   *Feature:* "Dependency Tracing." If a lawyer wants to cite Article 17, the Graph automatically pulls all related regulations (Art 269, 270) and Supreme Court interpretations (if added later).
    *   *Value:* Saves hours of cross-referencing.
2.  **Civic Tech (Citizen's Pocket Guide):**
    *   *Feature:* "Am I allowed to...?" Simple semantic search for citizens (e.g., "Can I open a shop anywhere?").
    *   *Value:* Democratizes legal access. The Graph makes complex "Legalese" navigable.

#### Risks & Mitigations
1.  **Temporal Drift (Amendments):**
    *   *Risk:* The Constitution changes. Training on 2015 data makes the model obsolete in 2025.
    *   *Mitigation:* The "Snapshot" approach we built (ignoring `AMENDED_BY` for training) is safe for historical analysis, but a Production App must implement **Graph Versioning** (Time Trees) to serve the *current* law.
2.  **Hallucination:**
    *   *Risk:* LLMs invent laws.
    *   *Mitigation:* Our RAG architecture is **Grounded**. We only show answers if we find a high-similarity Article in the Vector Database. If `score < 0.25`, we return "Not Found" instead of guessing.


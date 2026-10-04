class SourceRanker:

    SCORE_MAP = {
        "HIGH": 1.0,
        "MEDIUM": 0.7,
        "LOW": 0.3,
        "UNKNOWN": 0.0,
    }

    def rank(self, evaluated_sources):

        if not evaluated_sources:
            return []

        ranked_sources = []

        for source in evaluated_sources:

            authority = source.get("authority", "UNKNOWN").upper()
            relevance = source.get("relevance", "UNKNOWN").upper()
            usefulness = source.get("usefulness", "UNKNOWN").upper()

            authority_score = self.SCORE_MAP.get(authority, 0.0)
            relevance_score = self.SCORE_MAP.get(relevance, 0.0)
            usefulness_score = self.SCORE_MAP.get(usefulness, 0.0)

            # Relevance gets the highest importance because
            # a highly authoritative source is not useful
            # if it does not answer the research question.
            score = (
                authority_score * 0.35
                + relevance_score * 0.40
                + usefulness_score * 0.25
            )

            ranked_sources.append({
                "title": source.get("title", ""),
                "url": source.get("url", ""),
                "authority": authority,
                "relevance": relevance,
                "usefulness": usefulness,
                "score": round(score, 3),
                "reason": source.get("reason", ""),
            })

        ranked_sources.sort(
            key=lambda x: x["score"],
            reverse=True
        )

        for rank, source in enumerate(ranked_sources, start=1):
            source["rank"] = rank

        return ranked_sources
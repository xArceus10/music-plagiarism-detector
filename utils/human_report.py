def generate_human_report(uploaded_filename, matches, threshold=0.85):
    if not matches:
        return f"No similar tracks found for '{uploaded_filename}'. You are good to go!"

    top_match = matches[0]
    plagiarized = top_match['similarity_score'] >= threshold

    report = [f"Plagiarism Report for: '{uploaded_filename}'"]
    report.append(f"\nStatus: {'⚠️ Potential Plagiarism Detected' if plagiarized else '✅ No Critical Similarities Found'}")
    report.append("\nMost Similar Tracks:")

    for i, match in enumerate(matches, 1):
        sim = round(match['similarity_score'] * 100, 2)
        report.append(f"\n{i}. \"{match['title']}\"\n   - Similarity: {sim}%")
        if sim >= 0.85:
            report.append("   - Reason: Very close match in melody or rhythm")
        elif sim >= 0.7:
            report.append("   - Reason: Partial overlap in structure")
        else:
            report.append("   - Reason: Some general similarity")

    verdict = (
        f"\nFinal Verdict:\nYour track is highly similar to \"{top_match['title']}\" "
        f"(Similarity: {round(top_match['similarity_score'] * 100, 1)}%), which exceeds the plagiarism threshold "
        f"of {int(threshold * 100)}%.\nWe recommend revising your track before publishing."
        if plagiarized else
        "\nFinal Verdict:\nYour track appears original based on our analysis."
    )

    report.append(verdict)
    return "\n".join(report)

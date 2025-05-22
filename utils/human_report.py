def generate_human_report(uploaded_filename, matches, threshold=0.85):
    if not matches:
        return f"No similar tracks found for '{uploaded_filename}'. You are good to go!"

    report = [f"🎧 Plagiarism Report for: '{uploaded_filename}'"]
    report.append("\n📊 Most Similar Tracks:")

    potential_issues = []

    for i, match in enumerate(matches, 1):
        audio_sim = match.get('audio_score')
        lyrics_sim = match.get('lyrics_score')

        report.append(f"\n{i}. \"{match['title']}\"")

        if audio_sim is not None:
            sim_pct = round(audio_sim * 100, 2)
            report.append(f"   🎵 Audio Similarity: {sim_pct}%")
            if audio_sim >= 0.85:
                report.append("   - Reason: Very close match in melody or rhythm")
                potential_issues.append(f"{match['title']} (Audio: {sim_pct}%)")
            elif audio_sim >= 0.7:
                report.append("   - Reason: Partial overlap in audio structure")
            else:
                report.append("   - Reason: Some general audio similarity")

        if lyrics_sim is not None:
            sim_pct = round(lyrics_sim * 100, 2)
            report.append(f"   📝 Lyrics Similarity: {sim_pct}%")
            if lyrics_sim >= 0.85:
                report.append("   - Reason: Very close match in lyrical phrasing or meaning")
                potential_issues.append(f"{match['title']} (Lyrics: {sim_pct}%)")
            elif lyrics_sim >= 0.7:
                report.append("   - Reason: Partial overlap in themes or phrases")
            else:
                report.append("   - Reason: Some general lyrical similarity")

    # Final verdict
    if potential_issues:
        report.append("\n⚠️ Final Verdict:")
        report.append("Potential plagiarism detected based on the following matches:")
        for issue in potential_issues:
            report.append(f"   - {issue}")
        report.append("\nWe recommend reviewing or revising your track before publishing.")
    else:
        report.append("\n✅ Final Verdict:\nYour track appears original based on both audio and lyrics analysis.")

    return "\n".join(report)

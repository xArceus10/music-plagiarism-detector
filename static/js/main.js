document.getElementById("uploadForm").addEventListener("submit", async (e) => {
    e.preventDefault();

    const audio = document.getElementById("audio").files[0];
    const lyrics = document.getElementById("lyrics").files[0];
    if (!audio) return alert("Please upload an audio file!");

    const formData = new FormData();
    formData.append("audio", audio);
    if (lyrics) formData.append("lyrics", lyrics);

    document.getElementById("loading").style.display = "block";
    document.getElementById("result").style.display = "none";

    const res = await fetch("/analyze", { method: "POST", body: formData });
    const data = await res.json();

    document.getElementById("loading").style.display = "none";
    document.getElementById("result").style.display = "block";

    const { audio_score, lyrics_score, hybrid_score, decision } = data;

    updateBar("audio", audio_score);
    updateBar("lyrics", lyrics_score);
    updateBar("hybrid", hybrid_score);

    document.getElementById("decision").innerText = decision;
});

function updateBar(type, value) {
    const percent = (value * 100).toFixed(2) + "%";
    document.getElementById(`${type}-bar`).style.width = percent;
    document.getElementById(`${type}-val`).innerText = percent;
}

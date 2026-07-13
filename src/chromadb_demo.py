import chromadb
from chromadb.utils.embedding_functions.ollama_embedding_function import (
    OllamaEmbeddingFunction,
)
chroma_client = chromadb.Client()

collection = chroma_client.create_collection(
    name="demo_collection",
    embedding_function = OllamaEmbeddingFunction(
        model_name="nomic-embed-text-v2-moe",
        url="http://localhost:11434"
    )
)

collection.add(
    ids=[
        "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12",
        "13", "14", "15", "16", "17", "18", "19", "20", "21", "22",
        "23", "24",
    ],
    documents=[
        "Die Imaging Solutions AG ist ein Schweizer Hersteller, der sich auf modulare Systeme für die automatisierte",
        "Produktion von On -Demand -Fotodrucken, Layflat -Büchern und Leinwandrahmen spezialisiert hat. Seit 2003",
        "hat sich Imaging Solutions als einer der führe nden Anbieter von kundenspezifischen Lösungen für die",
        "Bildnachbearbeitungsindustrie weltweit etabliert.",
        "Deine Aufgaben :",
        "• Inbetriebnahme und Prüfung der Maschinen",
        "• Termingerechte Bereitstellung der Maschinen gemäss Kundenauftrag",
        "• Sicherstellen der Qualität und Funktionalität der Maschinen",
        "• Baugruppenprüfung nach Prüfvorschriften und Verfahrensanweisungen",
        "• Erstellen von Prüfdokumentationen und Checklisten",
        "• Aufbau von Entwicklungsbaugruppen und deren Inbetriebnahme und Prüfung",
        "• Mithilfe in der R&D Abteilung inkl. Tests von Baugruppen und Maschinen mit derer Dokumentation",
        "• Service-Unterstützung bei Telefonsupport und weltweiten Kundeneinsätzen",
        "(besonders während Peak -Season Nov.-Dez.)",
        "• Dein Profil :",
        "• Ausbildung als Elektroniker B/ Automatiker / Elektromechaniker oder Maschinenmechaniker E",
        "• Sehr gute Kenntnisse in Mechanik, Elektrik und Pneumatik",
        "• Kenntnisse in der Papierverarbeitung und Klebetechnik von Vorteil",
        "• Einige Jahre Berufserfahrung in einer ähnlichen Position",
        "• Gute Sprachkenntnisse in Deutsch (W+S), Englisch von Vorteil",
        "• Sehr gute MS-Office Kenntnisse besonders Word und Excel",
        "• Gute schriftliche Ausdrucksfähigkeit sowie die Fähigkeit, klare und strukturierte Texte (z. B.",
        "Prüfdokumentationen und technische Berichte) zu verfassen",
        "Sehr gute MS-Office Kenntnisse besonders Word und Excel",
    ]
)

results = collection.query(
    query_texts=[
        "Microsoft",
        "Am Computer Dokumente bearbeiten"
    ]
)

dummy = 1
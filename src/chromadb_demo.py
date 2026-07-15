# Make sure to run ollama and have the models used pulled to your machine before running!

from typing import cast
import chromadb
from chromadb.api.types import EmbeddingFunction
from chromadb.utils.embedding_functions.ollama_embedding_function import OllamaEmbeddingFunction

chroma_client = chromadb.PersistentClient(path="./chroma")

ollama_ef = OllamaEmbeddingFunction(
        url="http://localhost:11434",
        model_name="nomic-embed-text-v2-moe"
)
ollama_ef = cast(EmbeddingFunction, ollama_ef) # just so pylance doesn't freak out

collection = chroma_client.get_or_create_collection(
    name="demo_collection",
    embedding_function=ollama_ef,
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
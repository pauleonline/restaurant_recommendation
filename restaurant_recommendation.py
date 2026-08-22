"""
Restaurant Recommendation App
"""

import math
import time
import json
import pandas as pd
import streamlit as st
import pydeck as pdk

import googlemaps
from geopy.distance import geodesic as GD
from geopy.geocoders import Nominatim

from langchain_openrouter import ChatOpenRouter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
# from langchain.output_parsers import OutputFixingParser

##############################################################################
#  CONFIGURATION
##############################################################################

st.set_page_config(
    page_title="Restaurant Recommendation",
    page_icon="🍴", layout="wide", initial_sidebar_state="expanded",)

##############################################################################
#  STATIC CONFIG / CATALOGS
##############################################################################
OPENROUTER_MODELS = [
    "nvidia/nemotron-3-nano-30b-a3b:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "nvidia/nemotron-3-ultra-550b-a55b:free",
    "inclusionai/ling-3.0-flash:free",
    "openrouter/free",
]

BUSINESS_COLS = ["name", "rating", "review", "price", "distance", "address", "category_list", "id"]
HTML_COLS = [
    "name", "rating", "review", "price", "distance", "address",
    "beschreibung", "oeffnung", "link",]

PAGE_SIZE = 20     # Google Places liefert max. 20 Ergebnisse pro Seite

gmaps = googlemaps.Client(key=st.secrets["GOOGLE_MAPS_KEY"])
geolocator = Nominatim(user_agent="geoapiExercises")

st.session_state.setdefault("model", OPENROUTER_MODELS[0])
st.session_state.setdefault("business_df", None)
st.session_state.setdefault("glocation", None)
st.session_state.setdefault("location_orig", None)
model = st.session_state.model
business_df = st.session_state.business_df

class RestaurantInfo(BaseModel):
    beschreibung: str = Field(description="Küche, Empfehlungen/Spezialitäten, Ambiente, Besonderheiten – max. 500 Zeichen")
    link: str = Field(default="", description="Website-URL des Restaurants, falls bekannt, sonst leerer String")
    oeffnungszeiten: str = Field(default="", description="Öffnungszeiten, falls bekannt, sonst leerer String")

_base_parser = PydanticOutputParser(pydantic_object=RestaurantInfo)

description_prompt = ChatPromptTemplate.from_template(
    "Du bist ein erfahrener Restaurant Kritiker und Local Guide.\n"
    "Gib mir zu dem Restaurant {name} in {address} eine kurze, stichpunktartige "
    "Beschreibung: welche Küche(n) angeboten wird, welche Speisen/Spezialitäten "
    "besonders zu empfehlen sind, wie das Ambiente ist und eventuelle sonstige "
    "Besonderheiten. Ermittle auch die Öffnungszeiten und eine Website-URL, unter der die Speisekarte einsehbar ist."
    "Maximal 750 Zeichen. Falls Website-URL oder Öffnungszeiten "
    "nicht bekannt sind, lasse die entsprechenden Felder leer statt zu raten.\n\n"
    "{format_instructions}"
)

##############################################################################
#  DATA / RESOURCE LOADING
##############################################################################
@st.cache_resource(show_spinner=False)
def get_openrouter_model(model_name: str, temperature: float) -> ChatOpenRouter:
    """Erstellt (und cached) das LangChain-Chat-Modell für OpenRouter."""
    api_key = st.secrets["OPENROUTER_API_KEY"]
    if not api_key: raise ValueError("OPENROUTER_API_KEY not found in environment/.env")
    return ChatOpenRouter(model=model_name, temperature=temperature, api_key=api_key)

##############################################################################
#  LLM HELPERS
##############################################################################
def run_llm(query: str, llm_model, spinner_text: str) -> str:
    """Führt einen Prompt gegen das LLM aus und gibt den reinen Text zurück."""
    with st.spinner(spinner_text):
        try:
            prompt_template = ChatPromptTemplate.from_template("{user_query}")
            chain = prompt_template | llm_model | StrOutputParser()
            return chain.invoke({"user_query": query})
        except Exception as exc:
            st.error(f"Fehler beim Ausführen des Prompts: {exc}")
            return ""

def get_address(address: str) -> tuple:
    """Geocodiert eine Adresse, mit Fallback von Nominatim auf Google Maps."""
    try:
        location = geolocator.geocode(address)
        if location is None: raise ValueError("Nominatim lieferte keinen Treffer")
        return (location.latitude, location.longitude, location.address)
    except Exception:
        try:
            results = gmaps.geocode(address)
            if not results:  raise ValueError("Google Maps lieferte keinen Treffer")
            location = results[0]
            lat = location["geometry"]["location"]["lat"]
            lng = location["geometry"]["location"]["lng"]
            short_addr = ", ".join(c["short_name"] for c in location["address_components"][:4])
            return (lat, lng, short_addr)
        except Exception: return (0, 0, "")

def fetch_places(glocation: dict) -> pd.DataFrame:
    """Holt bis zu ANZ Restaurants von der Google Places Nearby API."""
    frames = []
    response = None
    num_pages = math.ceil(ANZ / PAGE_SIZE)

    for page in range(num_pages):
        if page == 0: response = gmaps.places_nearby(keyword=TERM, location=glocation, rank_by="distance")
        else:
            next_token = response.get("next_page_token") if response else None
            if not next_token: break  # keine weiteren Seiten verfügbar
            time.sleep(2)  # Google benötigt eine kurze Pause, bevor next_page_token gültig wird
            response = gmaps.places_nearby(keyword=TERM, location=glocation, rank_by="distance", page_token=next_token)
        frames.append(pd.DataFrame(response.get("results", [])))

    if not frames: return pd.DataFrame(columns=BUSINESS_COLS)
    return pd.concat(frames, ignore_index=True)

def get_restaurant_info(name: str, address: str, llm_model) -> RestaurantInfo:
    fixing_parser = _base_parser # oder OutputFixingParser.from_llm(parser=_base_parser, llm=llm_model)
    chain = description_prompt | llm_model | fixing_parser
    try:
        return chain.invoke({
            "name": name,
            "address": address,
            "format_instructions": _base_parser.get_format_instructions(),
        })
    except Exception as exc:
        st.warning(f"Beschreibung für '{name}' konnte nicht erzeugt werden: {exc}")
        return RestaurantInfo(beschreibung="", link="", oeffnungszeiten="")
    
def build_restaurant_context(df: pd.DataFrame) -> str:
    """Baut Kontext nur aus Zeilen mit vorhandener Beschreibung."""
    described = df[df["beschreibung"].fillna("").str.strip() != ""]
    lines = []
    for _, row in described.iterrows():
        lines.append(
            f"- {row['name']} | {row['address']} | Rating: {row['rating']} "
            f"({row['review']} Reviews) | {row['beschreibung'].strip()}"
        )
    return "\n".join(lines)
    
##############################################################################
#  SIDEBAR
##############################################################################
st.sidebar.title("⚙️ Engine Settings")

selected_model         = st.sidebar.selectbox("Model:", OPENROUTER_MODELS, index=0)
st.session_state.model = selected_model

temperature            = st.sidebar.slider("Creativity (Temperature)", 0.0, 1.0, 0.2, 0.1)
ANZ                    = st.sidebar.slider("Anzahl (Restaurants)", 1, 100, 40, 1)
TERM                   = st.sidebar.text_input("Suchbegriff", value="Restaurant")

try:
    llm_model = get_openrouter_model(selected_model, temperature)
except ValueError as exc:
    st.error(f"⚠️ {exc}. Bitte OPEN_ROUTER_API_KEY als Umgebungsvariable/.env-Eintrag setzen.")
    st.stop()

##############################################################################
#  MAIN INTERFACE
##############################################################################
st.title("🍴 Restaurant Recommendation")
st.caption("Best Restaurants in the nearby")

##############################################################################
#  Liste der Restaurants
##############################################################################

col1, col2, col3         = st.columns([6,2, 2]) 
with col1: location_orig = st.text_input('Adresse')
with col2: min_reviews   = st.number_input('Min. Reviews', value=20, min_value=0, step=1)
with col3: min_rating    = st.number_input('Min. Rating', value=4.5, min_value=0.0, max_value=5.0, step=0.1)

if st.button("Recommend Restaurants", type="primary"):
    if not location_orig:
        st.warning("Bitte eine Adresse eingeben.")
        st.stop()

    geocode_results = gmaps.geocode(location_orig)
    if not geocode_results:
        st.error(f"Adresse '{location_orig}' konnte nicht gefunden werden.")
        st.stop()

    glocation = geocode_results[0]["geometry"]["location"]
    st.session_state.glocation = glocation
    st.session_state.location_orig = location_orig

    business_df = fetch_places(glocation)
    if business_df.empty:
        st.warning("Keine Restaurants gefunden.")
        st.stop()

    business_df = business_df[(business_df["rating"] >= min_rating) & (business_df["user_ratings_total"] >= min_reviews)] 
    if business_df.empty:
        st.warning("Keine Restaurants erfüllen die Filterkriterien.")
        st.stop() 

    business_df = business_df.drop_duplicates(subset="place_id").reset_index(drop=True)[:ANZ]

    ###############################
    # umbenennen
    business_df["price"]   = business_df.get("price_level")
    business_df["review"]  = business_df.get("user_ratings_total")
    business_df["address"] = business_df.get("vicinity")
    business_df["id"]      = business_df.get("place_id")

    ###############################
    # berechnen
    business_df["category_list"] = business_df.get("types", pd.Series([[]] * len(business_df))).apply(
        lambda types: ", ".join(str(t) for t in types)[:40] )
    
    coords = business_df["address"].apply(get_address)
    business_df[["lat", "lon", "standard_addr"]] = pd.DataFrame(coords.tolist(), index=business_df.index)
    business_df["distance"] = business_df.apply(lambda r: int(GD((r.lat, r.lon), (glocation["lat"], glocation["lng"])).m), axis=1)
    business_df = business_df.sort_values(["distance", "standard_addr", "name"], ascending=True).reset_index(drop=True)

    business_df["beschreibung"] = ""
    business_df["link"]     = ""
    business_df["oeffnung"] = ""
    
    st.session_state.business_df = business_df
    st.session_state.glocation = glocation
    st.session_state.location_orig = location_orig

##############################################################################
#  PREFERENCE-BASED TOP-3 RECOMMENDATION
##############################################################################

if st.session_state.business_df is not None and not st.session_state.business_df.empty:
    st.divider()
    st.subheader("🔍 Empfehlung nach Präferenz")

    user_preference = st.text_input("Wonach suchst du?", 
        placeholder="z. B. 'romantisches Restaurant für ein Date' oder 'günstige italienische Küche'", )

    if st.button("Top 3 empfehlen"):
        if not user_preference: st.warning("Bitte eine Präferenz eingeben.")
        else:
            df = st.session_state.business_df

            if "beschreibung" not in df.columns or ( df["beschreibung"].fillna("").eq("").all()): 
                st.warning("Keine Restaurant-Beschreibungen vorhanden. "
                    "Bitte zuerst 'Mit Beschr.' aktivieren und neu suchen.")
            else:
                context = build_restaurant_context(df)
                prompt = (
                    "Du bist ein erfahrener Restaurant-Berater. Hier ist eine Liste von "
                    f"Restaurants mit Beschreibung:\n\n{context}\n\n"
                    f"Der Nutzer sucht: \"{user_preference}\"\n\n"
                    "Wähle die 3 am besten passenden Restaurants aus der Liste aus "
                    "(nur Namen, die exakt in der Liste stehen) und begründe jede Wahl "
                    "in 1-2 Sätzen anhand der Beschreibung. Antworte als nummerierte Liste "
                    "im Format:\n1. **Name** - Begründung")

                recommendation = run_llm(prompt, llm_model, "Suche passende Empfehlungen...")
                if recommendation: st.markdown(recommendation)

##############################################################################
#  DESCRIPTION SELECTION
##############################################################################
if st.session_state.business_df is not None and not st.session_state.business_df.empty:
    st.divider()
    st.subheader("📝 Beschreibungen abrufen")
    st.caption("Wähle aus, für welche Restaurants eine KI-Beschreibung erstellt werden soll.")

    display_df = st.session_state.business_df.copy()
    display_df.insert(0, "auswahl", False)

    edited_df = st.data_editor(
        display_df[["auswahl", "name", "rating", "review", "price", "distance", "address",
                    "beschreibung", "link", "oeffnung"]],
        column_config={
            "auswahl": st.column_config.CheckboxColumn("Auswahl", default=False),
            "beschreibung": st.column_config.TextColumn("Beschreibung", disabled=True),
            "link": st.column_config.TextColumn("Link", disabled=True),
            "oeffnung": st.column_config.TextColumn("Öffnungszeiten", disabled=True),
            "name": st.column_config.TextColumn("Name", disabled=True),
            "rating": st.column_config.NumberColumn("Rating", disabled=True),
            "review": st.column_config.NumberColumn("Reviews", disabled=True),
            "price": st.column_config.TextColumn("Preis", disabled=True),
            "distance": st.column_config.NumberColumn("Entfernung (m)", disabled=True),
            "address": st.column_config.TextColumn("Adresse", disabled=True),
        },
        disabled=["name", "rating", "review", "price", "distance", "address",
                  "beschreibung", "link", "oeffnung"],
        hide_index=True,
        use_container_width=True,
        key="business_editor",
    )

    selected_indices = edited_df.index[edited_df["auswahl"]].tolist()

    if st.button(f"Beschreibungen für Auswahl abrufen ({len(selected_indices)})", disabled=not selected_indices):
        for idx in selected_indices:
            row = st.session_state.business_df.loc[idx]
            with st.spinner(f"Analysiere {row['name']}..."):
                info = get_restaurant_info(row["name"], row["address"], llm_model)
            st.session_state.business_df.loc[idx, "beschreibung"] = info.beschreibung
            st.session_state.business_df.loc[idx, "link"] = info.link
            st.session_state.business_df.loc[idx, "oeffnung"] = info.oeffnungszeiten
        st.rerun()

##############################################################################
#  MAP
##############################################################################
if st.session_state.business_df is not None and not st.session_state.business_df.empty:
    st.divider()
    st.subheader("🗺️ Karte")

    nur_mit_beschreibung = st.checkbox('nur mit Beschreibung')

    def to_records(df: pd.DataFrame) -> list:
        """Erzwingt native Python-Typen für pydeck via JSON-Rundreise (NumPy-Skalare sind nicht JSON-serialisierbar)."""
        return json.loads(df.to_json(orient="records"))

    map_df = st.session_state.business_df.copy()
    if nur_mit_beschreibung: map_df = map_df[map_df['beschreibung']!=""]
    map_df = map_df[["name", "address", "distance", "lat", "lon"]]
    glocation = st.session_state.glocation

    search_point_df = pd.DataFrame([{
        "name": "📍 Gesuchte Adresse",
        "address": str(st.session_state.location_orig),
        "distance": "–",
        "lat": glocation["lat"],
        "lon": glocation["lng"],
    }])

    places_layer = pdk.Layer(
        "ScatterplotLayer",
        data=to_records(map_df),
        get_position=["lon", "lat"],
        get_fill_color=[255, 90, 60, 200],
        get_radius=35,
        pickable=True,
        stroked=True,
        get_line_color=[255, 255, 255],
        line_width_min_pixels=1,
    )

    search_layer = pdk.Layer(
        "ScatterplotLayer",
        data=to_records(search_point_df),
        get_position=["lon", "lat"],
        get_fill_color=[30, 120, 255, 230],
        get_radius=55,
        pickable=True,
        stroked=True,
        get_line_color=[255, 255, 255],
        line_width_min_pixels=2,
    )

    view_state = pdk.ViewState(
        latitude=float(glocation["lat"]),
        longitude=float(glocation["lng"]),
        zoom=14,
        pitch=0,
    )

    tooltip = {
        "html": "<b>{name}</b><br/>{address}<br/>Entfernung: {distance} m",
        "style": {"backgroundColor": "#1e293b", "color": "white"},
    }

    st.pydeck_chart(pdk.Deck(
        layers=[places_layer, search_layer],
        initial_view_state=view_state,
        tooltip=tooltip,
    ))
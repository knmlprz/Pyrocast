import folium
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import base64
from io import BytesIO
from PIL import Image
import requests
from math import radians, cos, sin, asin, sqrt

# --- Dane stacji IMGW (fragment) ---
stations_coords = {
    "Białystok": (53.1325, 23.1688),
    "Bielsko Biała": (49.8225, 19.0444),
    "Chojnice": (53.6956, 17.5572),
    "Częstochowa": (50.8118, 19.1203),
    "Elbląg": (54.1522, 19.4088),
    "Gdańsk": (54.3520, 18.6466),
    "Gorzów": (52.7368, 15.2288),
    "Hel": (54.6084, 18.8074),
    "Jelenia Góra": (50.9030, 15.7250),
    "Kalisz": (51.7611, 18.0910),
    "Kasprowy Wierch": (49.2325, 19.9847),
    "Katowice": (50.2599, 19.0216),
    "Kętrzyn": (54.0761, 21.3753),
    "Kielce": (50.8703, 20.6275),
    "Kłodzko": (50.4344, 16.6616),
    "Koło": (52.2027, 18.6387),
    "Kołobrzeg": (54.1755, 15.5830),
    "Koszalin": (54.1944, 16.1722),
    "Kozienice": (51.5833, 21.5500),
    "Kraków": (50.0647, 19.9450),
    "Krosno": (49.6883, 21.7700),
    "Legnica": (51.2100, 16.1619),
    "Lesko": (49.4700, 22.3300),
    "Leszno": (51.8401, 16.5740),
    "Lębork": (54.5500, 17.7500),
    "Lublin": (51.2465, 22.5684),
    "Łeba": (54.7574, 17.5503),
    "Łódź": (51.7592, 19.4560),
    "Mikołajki": (53.8022, 21.5813),
    "Mława": (53.1133, 20.3846),
    "Nowy Sącz": (49.6213, 20.7098),
    "Olsztyn": (53.7784, 20.4801),
    "Opole": (50.6751, 17.9213),
    "Ostrołęka": (53.0866, 21.5743),
    "Piła": (53.1511, 16.7383),
    "Platforma": (51.0000, 20.0000),  # WTF
    "Płock": (52.5469, 19.7069),
    "Poznań": (52.4064, 16.9252),
    "Przemyśl": (49.7833, 22.7833),
    "Racibórz": (50.0919, 18.2195),
    "Resko": (53.7706, 15.4300),
    "Rzeszów": (50.0413, 21.9990),
    "Sandomierz": (50.6820, 21.7484),
    "Siedlce": (52.1677, 22.2906),
    "Słubice": (52.3499, 14.5554),
    "Sulejów": (51.4800, 19.9300),
    "Suwałki": (54.1110, 22.9300),
    "Szczecin": (53.4285, 14.5528),
    "Szczecinek": (53.7100, 16.7000),
    "Śnieżka": (50.7363, 15.7390),
    "Świnoujście": (53.9090, 14.2478),
    "Tarnów": (50.0125, 20.9869),
    "Terespol": (52.0763, 23.6211),
    "Toruń": (53.0138, 18.5984),
    "Ustka": (54.5800, 16.8600),
    "Warszawa": (52.2297, 21.0122),
    "Wieluń": (51.2200, 18.5700),
    "Włodawa": (51.5464, 23.5495),
    "Wrocław": (51.1079, 17.0385),
    "Zakopane": (49.2992, 19.9496),
    "Zamość": (50.7239, 23.2517),
    "Zielona Góra": (51.9355, 15.5062)
}

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    lat1 = radians(lat1)
    lat2 = radians(lat2)
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    return 2 * R * asin(sqrt(a))

# --- Pobierz współrzędne adresu ---
def get_coordinates(address):
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": address, "format": "json", "limit": 1}
    headers = {"User-Agent": "MyGeocodingApp/1.0 (your@email.com)"}
    odp = requests.get(url, params=params, headers=headers)
    print("ODPOWIEDŹ NOMINATIM:", odp.json())
    if odp.status_code == 200 and odp.json():
        result = odp.json()[0]
        return float(result['lat']), float(result['lon'])
    else:
        return None, None

# --- Model smugowy Gaussa ---
def gaussian_plume(Q, u, wind_dir, H, x, y, z=0):
    theta = np.radians(wind_dir)
    x_wind = x * np.cos(theta) + y * np.sin(theta)
    y_wind = -x * np.sin(theta) + y * np.cos(theta)
    mask = x_wind > 0
    C = np.zeros_like(x_wind)
    sigma_y = 0.22 * x_wind[mask] * (1 + 0.0001 * x_wind[mask])**-0.5
    sigma_z = 0.20 * x_wind[mask]
    term1 = Q / (2 * np.pi * u * sigma_y * sigma_z)
    term2 = np.exp(-0.5 * (y_wind[mask] / sigma_y)**2)
    term3 = np.exp(-0.5 * ((z - H) / sigma_z)**2) + np.exp(-0.5 * ((z + H) / sigma_z)**2)
    C[mask] = term1 * term2 * term3
    return C

# --- Start programu ---
adres = "Plac defilad 1, Warszawa, Polska"
coords = get_coordinates(adres)

if coords:
    lat = coords[0]
    lon = coords[1]
else:
    lat = None
    lon = None

# Najbliższa stacja
closest_station = min(
    stations_coords.items(),
    key=lambda x: haversine(lat, lon, x[1][0], x[1][1])
)[0]
print(f"Najbliższa stacja IMGW: {closest_station}")

# Pobierz dane pogodowe z IMGW
try:
    response = requests.get("https://danepubliczne.imgw.pl/api/data/synop")
    data = response.json()
    station_data = next((s for s in data if s['stacja'].lower() == closest_station.lower()), None)
    u = float(station_data['predkosc_wiatru'])
    wind_dir = float(station_data['kierunek_wiatru'])
    print(f"Wiatr: {u} m/s, kierunek: {wind_dir}°")
except:
    print("Błąd pobierania danych IMGW.")
    exit(1)

# Parametry emisji
Q = 150.0
H = 20.0

# Siatka modelu
x = np.linspace(-1000, 1000, 300)
y = np.linspace(-1000, 1000, 300)
X, Y = np.meshgrid(x, y)

# Oblicz stężenie
C = gaussian_plume(Q, u, wind_dir, H, X, Y)

# Konwertuj na obraz RGBA
norm = mcolors.Normalize(vmin=0, vmax=np.max(C))
cmap = plt.cm.inferno
img_rgba = (cmap(norm(C)) * 255).astype(np.uint8)
img = Image.fromarray(img_rgba)
img_io = BytesIO()
img.save(img_io, format='PNG')
img_io.seek(0)
image_url = 'data:image/png;base64,' + base64.b64encode(img_io.getvalue()).decode()

# Rozmiar nakładki w stopniach (ok. 1 km = 0.009°)
lat_extent = 0.009 * (y.max() - y.min()) / 1000
lon_extent = 0.009 * (x.max() - x.min()) / 1000 / np.cos(np.radians(lat))
bounds = [[lat - lat_extent / 2, lon - lon_extent / 2],
          [lat + lat_extent / 2, lon + lon_extent / 2]]


# --- Mapa z nakładką i dynamiczną legendą ---
m = folium.Map(location=[lat, lon], zoom_start=14, tiles="CartoDB positron")

# Nakładka dymu
folium.raster_layers.ImageOverlay(
    image=image_url,
    bounds=bounds,
    opacity=0.6,
    name='Symulacja dymu'
).add_to(m)

# Znacznik źródła
folium.Marker(
    [lat, lon],
    popup=f"Źródło emisji<br>Wiatr: {u} m/s<br>Kierunek: {wind_dir}°",
    icon=folium.Icon(color='red')
).add_to(m)

# --- Generuj legendę kolorystyczną jako obrazek ---
fig, ax = plt.subplots(figsize=(1.0, 0.1))
fig.subplots_adjust(bottom=0.5)

cbar = plt.colorbar(
    plt.cm.ScalarMappable(norm=norm, cmap=cmap),
    cax=ax,
    orientation='horizontal'
)
cbar.set_label('Stężenie [g/m³]')
ax.set_title('Legenda natężenia', fontsize=9)

legend_buf = BytesIO()
plt.savefig(legend_buf, format='png', dpi=150, bbox_inches='tight', transparent=True)
legend_buf.seek(0)
legend_base64 = base64.b64encode(legend_buf.read()).decode('utf-8')
plt.close()

# --- Dodaj panel HTML z legendą i danymi pogodowymi ---
legend_html = f"""
<div style="
    position: fixed;
    top: 0px;
    left: 0;
    width: 100%;
    background-color: rgba(255, 255, 255, 0.95);
    padding: 8px;
    font-size: 15px;
    font-weight: 600;
    z-index: 10000;
    border-bottom: 1px solid #ccc;
    text-align: center;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
">
    Adres: {adres}
</div>

<div style="
    position: fixed;
    top: 60px;
    right: 10px;
    z-index: 9999;
    background-color: white;
    padding: 12px;
    border: 2px solid #ccc;
    border-radius: 8px;
    font-size: 13px;
    box-shadow: 2px 2px 6px rgba(0,0,0,0.3);
    line-height: 1.5;
">
    <b>Informacje pogodowe</b><br>
    Wiatr: <b>{u:.1f} m/s</b><br>
    Kierunek: <b>{wind_dir:.0f}°</b><br><br>
    <img src="data:image/png;base64,{legend_base64}" style="width:100%;"><br>
</div>
"""

m.get_root().html.add_child(folium.Element(legend_html))

# Kontrola warstw
folium.LayerControl().add_to(m)

# Zapisz mapę
map_path = "C:\\Users\\hp\\PycharmProjects\\Spaceshield\\symulacja_dymu_na_mapie.html"
m.save(map_path)
print(f"Zapisano mapę: {map_path}")


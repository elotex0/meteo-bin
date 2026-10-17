NODATA_U8 = 255
LEVELS = 254

ENCODING = {
    "t2m":         {"vmin": -35.0, "vmax": 45.0,   "transform": "linear", "unit": "°C"},
    "wind":        {"vmin": 0.0,   "vmax": 250.0,  "transform": "linear", "unit": "km/h"},
    "tp":          {"vmin": 0.0,   "vmax": 125.0,  "transform": "sqrt",   "unit": "mm"},
    "tp_acc":      {"vmin": 0.0,   "vmax": 500.0,  "transform": "sqrt",   "unit": "mm"},
    "cape_ml":     {"vmin": 0.0,   "vmax": 5000.0, "transform": "sqrt",   "unit": "J/kg"},
    "dbz_cmax":    {"vmin": -20.0, "vmax": 80.0,   "transform": "linear", "unit": "dBZ"},
    "cloud":       {"vmin": 0.0,   "vmax": 100.0,  "transform": "linear", "unit": "%"},
    "snow":        {"vmin": 0.0,   "vmax": 400.0,  "transform": "sqrt",   "unit": "cm"},
    "change_snow": {"vmin": -100.0, "vmax": 100.0, "transform": "ssqrt",  "unit": "cm"},
    "twater":      {"vmin": 0.0,   "vmax": 100.0,  "transform": "linear", "unit": "mm"},
    "snowfall":    {"vmin": 0.0,   "vmax": 6000.0, "transform": "linear", "unit": "m"},
    "ww":          {"vmin": 0.0,   "vmax": 254.0,  "transform": "raw",    "unit": ""},
}
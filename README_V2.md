
## 🗺️ ROADMAP CUBEXPRESS

### FASE 0: Fundamentos y Deuda Técnica (Crítico)

| ID | Tarea | Complejidad | Dependencias |
|----|-------|-------------|--------------|
| 0.1 | **Tests unitarios básicos** - pytest para cada módulo | Media | Ninguna |
| 0.2 | **Typing completo** - Añadir type hints faltantes y validar con mypy | Baja | Ninguna |
| 0.3 | **Docstrings consistentes** - Formato numpy/google en todas las funciones públicas | Baja | Ninguna |
| 0.4 | **CI/CD básico** - GitHub Actions para tests y linting | Media | 0.1 |
| 0.5 | **Manejo de errores robusto** - Excepciones específicas y mensajes claros | Media | Ninguna |

### FASE 1: Core Improvements (Splitting y Geometría)

| ID | Tarea | Complejidad | Dependencias |
|----|-------|-------------|--------------|
| 1.1 | **Splitting flexible** - No solo potencias de 2, división óptima según límite GEE | Media | Ninguna |
| 1.2 | **Detección lazy de dimensiones** - Obtener width/height/transform por chunks durante descarga | Media | Ninguna |
| 1.3 | **Full scene download** - Descargar escenas completas sin especificar edge_size | Alta | 1.1, 1.2 |
| 1.4 | **Caché de geometrías** - Almacenar transforms por tile/path-row para reutilizar | Media | 1.2 |
| 1.5 | **Validación de límites GEE** - Detectar y manejar límites de pixels/request dinámicamente | Media | 1.1 |

### FASE 2: Geometría Avanzada (Polígonos y CRS)

| ID | Tarea | Complejidad | Dependencias |
|----|-------|-------------|--------------|
| 2.1 | **Descarga por polígono (bbox)** - Bounding box del vector como ROI | Media | Ninguna |
| 2.2 | **Mask por polígono (conservativo)** - Pixels internos al polígono, resto nodata | Alta | 2.1 |
| 2.3 | **Mask por polígono (no conservativo)** - Pixels que intersectan el borde incluidos | Alta | 2.1 |
| 2.4 | **Multi-UTM handling** - Estrategia cuando polígono cruza zonas UTM | Alta | 2.1 |
| 2.5 | **Reproyección automática** - Opción de CRS de salida definido por usuario | Media | 2.4 |

### FASE 3: Formatos de Salida

| ID | Tarea | Complejidad | Dependencias |
|----|-------|-------------|--------------|
| 3.1 | **GeoTIFF con compresión configurable** - LZW, DEFLATE, ZSTD, etc. | Baja | Ninguna |
| 3.2 | **COG (Cloud Optimized GeoTIFF)** - Tiling interno y overviews | Media | 3.1 |
| 3.3 | **Formato PNG/JPG** - Para visualización (con normalización) | Baja | Ninguna |
| 3.4 | **Formato HDF5/NetCDF** - Para datacubes científicos | Media | Ninguna |
| 3.5 | **Formato Zarr** - Para cloud-native workflows | Media | Ninguna |
| 3.6 | **Return as xarray** - Opción de no escribir a disco, retornar xarray.DataArray | Media | Ninguna |
| 3.7 | **Return as numpy** - Retornar array raw sin metadata geoespacial | Baja | Ninguna |

### FASE 4: Performance y Escalabilidad

| ID | Tarea | Complejidad | Dependencias |
|----|-------|-------------|--------------|
| 4.1 | **Async downloads** - Migrar de ThreadPool a asyncio/aiohttp | Alta | Ninguna |
| 4.2 | **Progress granular** - Progreso por tile, no solo por imagen | Baja | Ninguna |
| 4.3 | **Retry con backoff** - Reintentos inteligentes en errores transitorios | Media | Ninguna |
| 4.4 | **Rate limiting** - Respetar límites de GEE automáticamente | Media | 4.3 |
| 4.5 | **Streaming merge** - Merge sin cargar todo en memoria | Alta | Ninguna |
| 4.6 | **Checkpointing** - Resumir descargas interrumpidas | Alta | Ninguna |

### FASE 5: API y Usabilidad

| ID | Tarea | Complejidad | Dependencias |
|----|-------|-------------|--------------|
| 5.1 | **Builder pattern** - API fluida tipo `CubeRequest().at(lon, lat).size(256).sensor("S2").download()` | Media | Ninguna |
| 5.2 | **CLI básico** - Comando `cubexpress download --sensor S2 --lon -0.09 --lat 51.5` | Media | Ninguna |
| 5.3 | **Configuración por archivo** - YAML/TOML para parámetros comunes | Baja | Ninguna |
| 5.4 | **Logging configurable** - Niveles, formato, archivo de salida | Baja | Ninguna |
| 5.5 | **Dry-run mode** - Mostrar qué se descargaría sin descargar | Baja | Ninguna |

---

## 🐛 BUGS Y MEJORAS DETECTADAS

| ID | Problema | Severidad | Archivo |
|----|----------|-----------|---------|
| B1 | `mss_table.py` duplica lógica de `cloud_utils.py` - código muerto | Baja | mss_table.py |
| B2 | Emojis con encoding incorrecto en prints (ðŸ"‚ en vez de 📂) | Baja | cloud_utils.py |
| B3 | `_apply_toa_if_needed` solo funciona para MSS pero el nombre es genérico | Media | request.py |
| B4 | Sin validación de `ee.Initialize()` antes de usar GEE | Media | Varios |
| B5 | `CONFIG` en config.py no se usa en ningún sitio | Baja | config.py |
| B6 | `merge_tifs` usa nodata=0 por defecto, puede ser problemático | Media | geospatial.py |
| B7 | Sin timeout en llamadas a GEE - puede colgar indefinidamente | Alta | cloud_utils.py, downloader.py |
| B8 | `_cache_key` usa MD5 (inseguro, aunque aquí no importa seguridad) | Baja | cache.py |
| B9 | Imports circulares potenciales entre módulos | Media | Varios |
| B10 | Sin cleanup de archivos temporales si falla el merge | Media | downloader.py |

---

## 📋 ORDEN DE IMPLEMENTACIÓN RECOMENDADO

```
SPRINT 1 (Fundamentos):
├── 0.1 Tests unitarios básicos
├── 0.5 Manejo de errores robusto
├── B4  Validación ee.Initialize()
├── B7  Timeouts en llamadas GEE
└── B1  Eliminar mss_table.py duplicado

SPRINT 2 (Core Splitting):
├── 1.1 Splitting flexible
├── 1.5 Validación límites GEE
├── 4.3 Retry con backoff
└── B6  Nodata configurable en merge

SPRINT 3 (Full Scenes):
├── 1.2 Detección lazy de dimensiones
├── 1.3 Full scene download
├── 1.4 Caché de geometrías
└── 4.2 Progress granular

SPRINT 4 (Formatos):
├── 3.1 GeoTIFF compresión configurable
├── 3.6 Return as xarray
├── 3.7 Return as numpy
└── 3.3 PNG/JPG output

SPRINT 5 (Polígonos):
├── 2.1 Descarga por polígono (bbox)
├── 2.2 Mask conservativo
├── 2.3 Mask no conservativo
└── 2.4 Multi-UTM handling

SPRINT 6 (Polish):
├── 3.2 COG output
├── 5.1 Builder pattern API
├── 5.2 CLI básico
└── 0.4 CI/CD
```

---


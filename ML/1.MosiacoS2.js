/**
*@autor   Andres LLanos
*
*/    
// llama la grilla con los tiles de Sentinel-2 y la almacena en la variable carta 
var carta = ee.FeatureCollection('projects/ee-monitoreo2024/assets/GridS2');

// Define la variables Grid, year, y maxCloud
//Filtra la información a partir de estos parametros
var Grid = '18NXH';
var year =  '2025';
var maxCloud = 20;

// Crea la variable con el identificador de la carta para poder desplegar en el mapa
var carta = carta.filter(ee.Filter.eq('Nombre', Grid));
var empty = ee.Image().byte();
var outline = empty.paint
    (
      {
      featureCollection: carta,
      color: 1,
      width: 2
    }
  );
// Adiciona la grilla al mapa
Map.addLayer(outline, {palette: 'FF0000'}, Grid, true);

// Filtra la coleccion de Sentinel-2 con estos metadados
var s2Sr = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
             .filter(ee.Filter.eq('MGRS_TILE', Grid))
             .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', maxCloud))

// Carga la colección de imágenes que estiman la probabilidad de nubes en cada píxel de imágenes Sentinel-2.
var s2Clouds = ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')

// Define las fechas de inicio y final para conformar el mosaico de mediana 
var START_DATE = ('2025-02-01');
var END_DATE = ('2025-02-28');
var MAX_CLOUD_PROBABILITY = maxCloud;

// Función para enmascarar las nubes en una imagen Sentinel-2 usando la capa de probabilidad.
function maskClouds(img) {
  var clouds = ee.Image(img.get('cloud_mask')).select('probability');
  var isNotCloud = clouds.lt(MAX_CLOUD_PROBABILITY);
  return img.updateMask(isNotCloud);
}

// Filtra la collección  de entrada por los rangos de datos y de region.
var criteria = ee.Filter.and(
               ee.Filter.bounds(carta),
               ee.Filter.date(START_DATE, END_DATE));
        s2Sr = s2Sr.filter(criteria);
    s2Clouds = s2Clouds.filter(criteria);

// Une S2_SR con cloud probability para añadir la cloud mask.
var s2SrWithCloudMask = ee.Join.saveFirst('cloud_mask').apply(
    {      primary: s2Sr,
      secondary: s2Clouds,
      condition:
      ee.Filter.equals({leftField: 'system:index', rightField: 'system:index'})
    }
);

print(s2SrWithCloudMask)

// Crea la variable y Aplica la función maskClouds a cada imagen de la colección s2SrWithCloudMask,
//calcula la mediana para reducir ruido temporal, y recorta el resultado al área definida por carta.
var s2CloudMasked = ee.ImageCollection(s2SrWithCloudMask)
                      .map(maskClouds)
                      .median()
                      .clip(carta);
                      
// Crea el mosaico con las bandas requeridas y lo almacena
var img = s2CloudMasked.select('B2', 'B3', 'B4', 'B5', 'B8', 'B11', 'B12');

// Define la visualización
var rgbVis = {
              min: 280,
              max: 1180, 
              gamma: 1,
              bands: ['B4']
              };   

// Adiciona al mapa
Map.addLayer(img.clip(carta), rgbVis,'Red' + Grid + '-' + year );

var rgbVis2 = {
    min: 401.59,
    max: 1365.9,
    bands: ['B4', 'B3', 'B2'],
  };
Map.addLayer(img.clip(carta), rgbVis2, 'RGB-' + Grid + '-' + year);

var rgbVis3 = {
    min: 900,
    max: 4000,
    gamma: 1,
    bands: ['B3'],
 };
Map.addLayer(img.clip(carta), rgbVis3, 'Green' + Grid + '-' + year, false);

var rgbVis3 = {
    min: 160,
    max: 3500,
    gamma: 1,
    bands: ['B11'],
 };
Map.addLayer(img.clip(carta), rgbVis3, 'SWIR1' + Grid + '-' + year, false);

var rgbVis3 = {
    min: 90,
    max: 3500,
    gamma: 1,
    bands: ['B12'],
 };
Map.addLayer(img.clip(carta), rgbVis3, 'SWIR2' + Grid + '-' + year, false);


// Calcula indices de vegetacion 
var ndvi = img.normalizedDifference(['B4','B8']).rename('NDVI');        // NDVI
var sr83 = img.select('B8').divide(img.select('B3')).rename('SR83');    // SR (Simple Ratio) 
var dvi = img.select('B8').divide(img.select('B4')).rename('DVI');      // DIFFERENCE VEGETATION INDEX 

var savi = img.expression('1.5 * (NIR - RED) / (0.5 + NIR + RED)',      // SOIL ADJUSTED VEGETATION INDEX
        {
        'RED': img.select('B4'),         // RED
        'NIR': img.select('B8')          // NIR  
         }).rename('SAVI');           

// Adiciona los indices al mapa con la visualizacion 
Map.addLayer(ndvi,{min:-0.8, max:-0.3} , 'ndvi', false)
Map.addLayer(savi,{min:0.25, max:1.2} , 'savi', false)
Map.addLayer(sr83,{min:1.4, max:8} , 'sr83', false)
Map.addLayer(dvi,{min:1.5, max:13} , 'dvi', false)

// Crea el satck con la banda4 el indice savi y dvi
var imgTest = img.select('B4').addBands(savi).addBands(dvi)
print(imgTest)

//Exporta al drive 
Export.image.toDrive({
  image: imgTest,
  description: Grid + '-' + year,
  scale: 10,
  crs: 'EPSG:4326',
  folder: 'GEE_Exports',
  maxPixels: 1e13,
  region: carta
});

// Exporta al cloud asset
Export.image.toAsset(
      {
        'image': imgTest,
        'description': Grid + '-' + year,
        'assetId': 'projects/ee-monitoreo2024/assets/' + Grid + '-' + year,   //**//
        'scale': 10,
        'crs': 'EPSG:4326',
        'region': carta.geometry().bounds(),
        'pyramidingPolicy': {
            '.default': 'mode'
        },
        'maxPixels': 1e13,
      }
);  

var carta = ee.FeatureCollection('projects/ee-monitoreo2024/assets/GridS2');

var Grid = '18NXH';
var year =  '2025';
var maxCloud = 20;

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
Map.addLayer(outline, {palette: 'FF0000'}, Grid, true);

var s2Sr = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
             .filter(ee.Filter.eq('MGRS_TILE', Grid))
             .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', maxCloud))
             
var s2Clouds = ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')
                 
var START_DATE = ('2025-02-01');
var END_DATE = ('2025-02-28');
var MAX_CLOUD_PROBABILITY = maxCloud;

function maskClouds(img) {
  var clouds = ee.Image(img.get('cloud_mask')).select('probability');
  var isNotCloud = clouds.lt(MAX_CLOUD_PROBABILITY);
  return img.updateMask(isNotCloud);
}

// Filter input collections by desired data range and region.
var criteria = ee.Filter.and(
               ee.Filter.bounds(carta),
               ee.Filter.date(START_DATE, END_DATE));
        s2Sr = s2Sr.filter(criteria);
    s2Clouds = s2Clouds.filter(criteria);

// Join S2 SR with cloud probability dataset to add cloud mask.
var s2SrWithCloudMask = ee.Join.saveFirst('cloud_mask').apply(
    {      primary: s2Sr,
      secondary: s2Clouds,
      condition:
      ee.Filter.equals({leftField: 'system:index', rightField: 'system:index'})
    }
);

print(s2SrWithCloudMask)

var s2CloudMasked = ee.ImageCollection(s2SrWithCloudMask)
                      .map(maskClouds)
                      .median()
                      .clip(carta);
                      
var img = s2CloudMasked.select('B2', 'B3', 'B4', 'B5', 'B8', 'B11', 'B12');

var rgbVis = {
              min: 280,
              max: 1180, 
              gamma: 1,
              bands: ['B4']
              };   

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


//BIOMASS - VEGETATION
var ndvi = img.normalizedDifference(['B4','B8']).rename('NDVI');        // NDVI
var sr83 = img.select('B8').divide(img.select('B3')).rename('SR83');    // SR (Simple Ratio) 
var dvi = img.select('B8').divide(img.select('B4')).rename('DVI');      // DIFFERENCE VEGETATION INDEX 

var savi = img.expression('1.5 * (NIR - RED) / (0.5 + NIR + RED)',      // SOIL ADJUSTED VEGETATION INDEX
        {
        'RED': img.select('B4'),         // RED
        'NIR': img.select('B8')          // NIR  
         }).rename('SAVI');           

Map.addLayer(ndvi,{min:-0.8, max:-0.3} , 'ndvi', false)
Map.addLayer(savi,{min:0.25, max:1.2} , 'savi', false)
Map.addLayer(sr83,{min:1.4, max:8} , 'sr83', false)
Map.addLayer(dvi,{min:1.5, max:13} , 'dvi', false)

var imgTest = img.select('B4').addBands(savi).addBands(dvi)
print(imgTest)


Export.image.toDrive({
  image: imgTest,
  description: Grid + '-' + year,
  scale: 10,
  crs: 'EPSG:4326',
  folder: 'GEE_Exports',
  maxPixels: 1e13,
  region: carta
});


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

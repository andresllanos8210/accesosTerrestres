var imageCollection = ee.ImageCollection("projects/planet-nicfi/assets/basemaps/americas");

var carta = ee.FeatureCollection('projects/ee-monitoreo2024/assets/GridTest')
             
var Grid = '18NXH-1';
var year =  '2025';

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

var nicfi = ee.ImageCollection('projects/planet-nicfi/assets/basemaps/americas');
var START_DATE = ('2025-01'); 
var basemap = nicfi.filter(ee.Filter.date(START_DATE)).first().clip(carta);

var vis =
    {
    bands:['R','G','B'],
    min:94,
    max:1150,
    gamma:0.74
    };
Map.addLayer(basemap, vis, 'rgb', true);


var rVis = {
              min: 190,
              max: 850, 
              gamma: 1,
              bands: ['R']
              };   

Map.addLayer(basemap, rVis,'Red' + Grid + '-' + year );

//BIOMASS - VEGETATION
var ndvi = basemap.normalizedDifference(['R','N']).rename('NDVI');        // NDVI
var sr83 = basemap.select('N').divide(basemap.select('G')).rename('SR83');    // SR (Simple Ratio) 
var dvi = basemap.select('N').divide(basemap.select('R')).rename('DVI');      // DIFFERENCE VEGETATION INDEX 

var savi = basemap.expression('1.5 * (NIR - RED) / (0.5 + NIR + RED)',      // SOIL ADJUSTED VEGETATION INDEX
        {
        'RED': basemap.select('R'),         // RED
        'NIR': basemap.select('N')          // NIR  
         }).rename('SAVI');           

Map.addLayer(ndvi,{min:-0.8, max:-0.4} , 'ndvi', false)
Map.addLayer(savi,{min:0.6, max:1.3} , 'savi', false)
Map.addLayer(sr83,{min:3.3, max:9} , 'sr83', false)
Map.addLayer(dvi,{min:2, max:16} , 'dvi', false)

var imgTest = basemap.select('R').addBands(savi).addBands(dvi)
print(imgTest)

Export.image.toDrive({
  image: imgTest.toFloat(),
  description: Grid + '-' + year,
  scale: 5,
  crs: 'EPSG:4326',
  folder:'GEE_Exports',
  maxPixels: 1e13,
  region: carta
});


Export.image.toAsset(
      {
        'image': imgTest.toFloat(),
        'description': Grid + '-' + year,
        'assetId': 'projects/ee-monitoreo2024/assets/' + Grid + '-' + year,  //**//
        'scale': 5,
        'crs': 'EPSG:4326',
        'region': carta.geometry().bounds(),
        'pyramidingPolicy': {
            '.default': 'mode'
        },
        'maxPixels': 1e13,
      }
);  

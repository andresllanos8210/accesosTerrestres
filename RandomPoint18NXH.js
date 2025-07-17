var carta = ee.FeatureCollection('projects/ee-monitoreo2024/assets/GridS2');

var Grid = '18NXH';
var year =  '2025';
var Version = '2';
var carta = carta.filter(ee.Filter.eq('Nombre', Grid));

var empty = ee.Image().byte();
var outline = empty.paint
    ({
      featureCollection: carta,
      color: 1,
      width: 2
    });
Map.addLayer(outline, {palette: 'FF0000'}, Grid, true);

// var image = ee.Image('projects/ee-monitoreo2024/assets/Vias_TrainWGS84')
var image = ee.Image('projects/ee-monitoreo2024/assets/Vias18NXH')
Map.addLayer(image, {min:1, max:2, palette:['white','black']}, 'Vias', true);

var count = {
  1: 100, 
  2: 600
  };

print(count)

var stratified = ee.Dictionary(count)
    .map(function(klass,count)
    {
    klass = ee.Number.parse(klass)
    var masked = image.updateMask(image.eq(klass))
    return masked.addBands(ee.Image.pixelLonLat()).sample(
      {
      region: carta, 
      numPixels: 1000000, 
      seed: klass
      })
    //.randomColumn('x')
    //.sort('x')
    .limit(ee.Number(count).min(10000))
    .map(function(f)
      {
      var location = ee.Geometry.Point([f.get('longitude'),f.get('latitude')])
      return ee.Feature(location,f.toDictionary())
      })
    }).values()
stratified = ee.FeatureCollection(stratified).flatten(); 

var stratifiedVis = stratified.style(
  {
  width: 1,
  fillColor: 'white',  
  lineType: 'dotted',
  pointSize: 3,
  pointShape: 'circle'
  });
  
Map.addLayer(stratifiedVis, null,  'Random points');

print('Total Random points', stratified.size());

//Exportar como Asset
Export.table.toAsset(
    {
     collection: stratified,
     description:'RandomPoints-' + Grid + '-' + year + '-' + Version,
     assetId:'projects/ee-monitoreo2024/assets/Puntos-' + Grid + '-' + year + '-' + Version
    });

//Exportar como CSV a Google Drive
Export.table.toDrive(
    {
    collection: stratified,
    description:'RandomPoints-' + Grid + '-' + year + '-' + Version,
    folder: 'GEE_Exports',
    fileNamePrefix: 'Puntos-' + Grid + '-' + year + '-' + Version,
    fileFormat: 'CSV'
    });      

// Llama la grilla con los tiles de Sentinel-2 y la almacena en la variable carta 
var carta = ee.FeatureCollection('projects/ee-monitoreo2024/assets/GridS2');

// Define la variables Grid, year, y maxCloud
//Filtra la información a partir de estos parametros
var Grid = '18NXH';
var year =  '2025';
var Version = '2';  //Version

// Crea la variable con el identificador de la carta para poder desplegar en el mapa
var carta = carta.filter(ee.Filter.eq('Nombre', Grid));
var empty = ee.Image().byte();
var outline = empty.paint
    ({
      featureCollection: carta,
      color: 1,
      width: 2
    });
// Adiciona la grilla al mapa
Map.addLayer(outline, {palette: 'FF0000'}, Grid, true);

// Llama la informacion base de accesos terrestres de IDEAM en formato raster y despliega en el mapa
var image = ee.Image('projects/ee-monitoreo2024/assets/Vias18NXH')
Map.addLayer(image, {min:1, max:2, palette:['white','black']}, 'Vias', true);

// Función para generar puntos aleatorios
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
// Visualiación de puntos 
var stratifiedVis = stratified.style(
  {
  width: 1,
  fillColor: 'white',  
  lineType: 'dotted',
  pointSize: 3,
  pointShape: 'circle'
  });
  
// Adiciona puntos en el mapa
Map.addLayer(stratifiedVis, null,  'Random points');

print('Total Random points', stratified.size());

//Exporta la tabla de puntos al cloud Asset
Export.table.toAsset(
    {
     collection: stratified,
     description:'RandomPoints-' + Grid + '-' + year + '-' + Version,
     assetId:'projects/ee-monitoreo2024/assets/Puntos-' + Grid + '-' + year + '-' + Version
    });

//Exporta la tabla de puntos al Google Drive en formato CSV
Export.table.toDrive(
    {
    collection: stratified,
    description:'RandomPoints-' + Grid + '-' + year + '-' + Version,
    folder: 'GEE_Exports',
    fileNamePrefix: 'Puntos-' + Grid + '-' + year + '-' + Version,
    fileFormat: 'CSV'
    });      

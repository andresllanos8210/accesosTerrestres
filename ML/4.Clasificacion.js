// Llama la grilla con los tiles de Sentinel-2 y la almacena en la variable carta 
var carta = ee.FeatureCollection('projects/ee-monitoreo2024/assets/GridS2');

// Define la variables Grid, year, y maxCloud
//Filtra la información a partir de estos parametros
var Grid = '18NXH';
var year =  '2025';
var Version = '1';         //Numero de Version 
var puntos = 'puntos';     //Vesrion de Puntos o Muestreo     
var maxCloud = 20;         //Maxima cobertura de nubes de S2

// Crea la variable con el identificador de la carta para poder desplegar en el mapa
carta = carta.filter(ee.Filter.eq('Nombre', Grid));
var empty = ee.Image().byte();
var outline = empty.paint
    ({
      featureCollection: carta,
      color: 1,
      width: 2
    });
// Adiciona la grilla al mapa
Map.addLayer(outline, {palette: 'FF0000'}, Grid, true);

// Nombre del archivo de salida
var fileName = 'Vias-' + Grid + '-' + year + '-' + Version;

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
    min: 170,
    max: 1022,
    bands: ['B4', 'B3', 'B2'],
  };
Map.addLayer(img.clip(carta), rgbVis2, 'RGB-' + Grid + '-' + year);

// Calcula indices de vegetacion
var ndvi = img.normalizedDifference(['B4','B8']).rename('NDVI');        // NDVI
var sr83 = img.select('B8').divide(img.select('B3')).rename('SR83');    // SR (Simple Ratio) 
var dvi = img.select('B8').divide(img.select('B4')).rename('DVI');      // DIFFERENCE VEGETATION INDEX 

var savi = img.expression('1.5 * (NIR - RED) / (0.5 + NIR + RED)',      // SOIL ADJUSTED VEGETATION INDEX
        {
        'RED': img.select('B4'),         // RED
        'NIR': img.select('B8')          // NIR  
         }).rename('SAVI');
         
var evi = img.expression(
    'float (2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1)))',   //EVI
    // "2.5* ((B8 – B4) / (B8 + 6 * B4 – 7.5 * B2 + 1))"  
    //   G * ((NIR – R) / (NIR + C1 * R – C2 * B + L)) 
    {
     'BLUE':  img.select('B2'),    
     'RED':   img.select('B4'),   
     'NIR':   img.select('B8')     
    });  
    
var evi2 = img.expression('(2.5 * ((NIR - RED) / (NIR + (2.4 * RED) + 1)))',  //EVI2
        {
        'RED': img.select('B4'),     
        'NIR': img.select('B8')         
        })    
        
var msavi = img.expression(
      '((2*NIR) + 1 - (((2*NIR+1)**(2)) - (8*(NIR-RED)))**(1/2))  / 2',       //MSAVI 
      {
       'NIR':   img.select('B8'),      
       'RED':   img.select('B4')    
      });
      
var gosavi = img.expression(                                                  //MSAVI 
      '(NIR - GREEN) / (NIR + GREEN + 0.16)',
      {
      'GREEN': img.select('B3'),     
      'RED':   img.select('B4'),       
      'NIR':  img.select('B8')        
      }); 
      
var ndvi2 = img.expression('(NIR - SWIR) / (NIR + SWIR)',            // NORMALIZED DIFERENCE WATER INDEX VEGETATION
          {
          'NIR':  img.select ('B8'),  
          'SWIR': img.select ('B11')
          })             

// Definición de filtros morfologicos
// Kernel Sobel - dirección horizontal (Gx)
var sobelKernelX = ee.Kernel.fixed(3, 3, [
  [-1,  0,  1],
  [-2,  0,  2],
  [-1,  0,  1]
]);

// Kernel Sobel - dirección vertical (Gy)
var sobelKernelY = ee.Kernel.fixed(3, 3, [
  [-1, -2, -1],
  [ 0,  0,  0],
  [ 1,  2,  1]
]);

// Banda donde se aplicaran los filtros
var redBand = img.select('B4');

// Gradiente horizontal (bordes verticales)
var gradX = redBand.convolve(sobelKernelX).rename('Sobel_X');

// Gradiente vertical (bordes horizontales)
var gradY = redBand.convolve(sobelKernelY).rename('Sobel_Y');

// Magnitud del gradiente: sqrt(Gx² + Gy²)
var sobelMagnitude = gradX.pow(2).add(gradY.pow(2)).sqrt().rename('Sobel_Mag');

// Adiciona los filtros de gradiente al mapa
Map.addLayer(gradX.clip(carta), {min: -400, max: 400, palette: ['blue', 'white', 'red']}, 'Sobel X (bordes verticales)', false);
Map.addLayer(gradY.clip(carta), {min: -400, max: 400, palette: ['blue', 'white', 'red']}, 'Sobel Y (bordes horizontales)', false);
Map.addLayer(sobelMagnitude.clip(carta), {min: 0, max: 600, palette: ['black', 'white']}, 'Sobel Magnitude', false);

// Define el valor de kernel de 3x3
var kernel3 = ee.Kernel.square({radius: 1});

// Define filtro de erosión
var erosion = sobelMagnitude.reduceNeighborhood({
  reducer: ee.Reducer.min(),
  kernel: kernel3
}).rename('Erosion');

// Define filtro de dilatacion
var dilation = sobelMagnitude.reduceNeighborhood({
  reducer: ee.Reducer.max(),
  kernel: kernel3
}).rename('Dilation');

// Define filtro de apertura
var opening = erosion.reduceNeighborhood({
  reducer: ee.Reducer.max(),
  kernel: kernel3
}).rename('Opening');

// Define filtro de diltacion
var closing = dilation.reduceNeighborhood({
  reducer: ee.Reducer.min(),
  kernel: kernel3
}).rename('Closing');

// Adiciona los filtros morfologicos al mapa
Map.addLayer(sobelMagnitude.clip(carta), {min: 0, max: 1, palette: ['black', 'yellow']}, 'Original', false);
Map.addLayer(erosion.clip(carta), {min: 40, max: 1500, palette: ['black', 'blue']}, 'Erosion', false);
Map.addLayer(dilation.clip(carta), {min: 70, max: 2400, palette: ['black', 'green']}, 'Dilatación', false);
Map.addLayer(opening.clip(carta), {min: 30, max: 1300, palette: ['black', 'orange']}, 'Opening', false);
Map.addLayer(closing.clip(carta), {min: 80, max: 2400, palette: ['black', 'red']}, 'Closing', false);

// Une tres colecciones de entrenamiento (via, noVia, table) 
// en una sola llamada classNames, combinando todos sus elementos
var classNames = via.merge(noVia).merge(table);

// Construye una imagen multibanda llamada finalImage combinando bandas espectrales e índices derivados
var finalImage = img.addBands(ndvi.rename('NDVI'))  
                        .addBands(sr83.rename('SR83'))
                        .addBands(dvi.rename('DVI'))
                        .addBands(savi.rename('SAVI'))
                        .addBands(evi.rename('EVI'))
                        .addBands(evi2.rename('EVI2'))
                        .addBands(msavi.rename('MSAVI'))
                        .addBands(gosavi.rename('GOSAVI'))
                        .addBands(ndvi2.rename('NDVI2'))
                        .addBands(gradX.rename('GRADX'))
                        .addBands(gradY.rename('GRADY'))
                        .addBands(erosion.rename('EROSION'))
                        .addBands(opening.rename('OPENING'))
                        .addBands(dilation.rename('DILATION'))
                        .addBands(closing.rename('CLOSING'))
                        .addBands(sobelMagnitude.rename('MAGNITUDE'))

// Define la lista de bandas que se usarán en el análisis o clasificación. Incluye bandas espectrales (B2–B12), 
//índices de vegetación (NDVI, EVI, etc.) y transformaciones morfológicas (CLOSING, DILATION, etc)

var bands = ['B2', 'B3', 'B4', 'B5', 'B8', 'B11', 'B12', 'NDVI', 'SR83', 'DVI', 'SAVI', 
             'EVI', 'EVI2', 'MSAVI', 'GOSAVI', 'NDVI2', 'CLOSING', 'DILATION'/ 'GRADX', 
             'GRADY', 'EROSION', 'DILATION', 'OPENING', 'CLOSING', 'MAGNITUDE']

// - Extrae muestras de la imagen finalImage usando las bandas definidas en bands.
// - Usa las regiones de entrenamiento (classNames) para etiquetar cada muestra con la propiedad 'b1'.
// - Define la escala espacial a 10 metros por píxel.
// - Optimiza el procesamiento con tileScale: 16 (divide en bloques más grandes).
// - Agrega una columna aleatoria llamada 'random' para facilitar particiones (por ejemplo, entrenamiento vs validación).
var samples =  finalImage.select(bands).sampleRegions(
      {
        collection:classNames,
        properties: ['b1'],
        scale: 10,
      //projection: projection,
        tileScale: 16
      }).randomColumn('random');

var split = 0.8;                                                     // 80% para entrenamiento 20% para pruebas
var training = samples.filter(ee.Filter.lt('random', split));        // Crear subconjuntos de datos de entrenamiento
var testing = samples.filter(ee.Filter.gte('random', split));        // Crear subconjuntos de datos de prueba             
                                       
//Algoritmo de clasificacion de random forest 
var classifier = ee.Classifier.smileRandomForest(
    {
     'numberOfTrees':    50,
    // 'variablesPerSplit':   7,
    // 'minLeafPopulation':   1
    }).train(
      {
      features: training,
      classProperty: 'b1',
      inputProperties: bands
      }
    );
                          
// Se ejecuta el clasificador 
var classified = finalImage.select(bands).classify(classifier);

//Visualizar por clase con diferente color
var clase1 = table.filter(ee.Filter.eq('b1', 1));
var clase2 = table.filter(ee.Filter.eq('b1', 2));

Map.addLayer(clase1, {color: 'red'}, 'Vias');
Map.addLayer(clase2, {color: 'yellow'}, 'No Vias');

// Muestra la clasificacion sin corregir en el mapa  
Map.addLayer(
        classified.reproject('EPSG:4326', null, 10),
        {
        palette:
        [
         'black','white'
        ],
          min: 1,
          max: 2,
      },
        'classification-' + Grid + '-' + year
  );                        
  
var test = classified.sampleRegions({
  collection: table,
  properties: ['b1'],
  scale: 10,
  tileScale: 16,
  });

var testConfusionMatrix = test.errorMatrix('b1', 'classification')
print('Confusion Matrix', testConfusionMatrix);
print('Test Accuracy', testConfusionMatrix.accuracy());

// Mostrar la matriz de confusión
print('Matriz de Confusión', testConfusionMatrix);

// Calcular métricas básicas de desempeño
var accuracy = testConfusionMatrix.accuracy();
var precision = ee.Array(testConfusionMatrix.consumersAccuracy());
var recall = ee.Array(testConfusionMatrix.producersAccuracy());
var kappa = testConfusionMatrix.kappa();

// Calcular F1 Score por clase: 2 * (P * R) / (P + R)
var f1 = precision.multiply(recall).multiply(2).divide(precision.add(recall));

// Convertir arrays a listas para visualización
var precisionList = precision.toList();
var recallList = recall.toList();
var f1List = f1.toList();

// Mostrar resultados
print('Exactitud (Accuracy):', accuracy);
print('Precisión por clase:', precisionList);
print('Sensibilidad (Recall) por clase:', recallList);
print('F1 Score por clase:', f1List);
print('Índice Kappa:', kappa);

var explanation = classifier.explain();
print('Importancia de variables', classifier.explain())

// - Visualiza la importancia de variables del modelo Random Forest (importance) como gráfico de barras.
// - Evalúa la precisión del modelo variando el número de árboles (numTrees) y muestra un gráfico para 
//   encontrar el óptimo.
// - Explora combinaciones de numberOfTrees y bagFraction, generando una colección (resultFc) con la precisión
//   de cada configuración.
var importance = ee.Dictionary(explanation.get('importance'));
var feature = ee.Feature(null, importance);
var chart = ui.Chart.feature.byProperty(feature)
  .setChartType('ColumnChart')
  .setOptions({
    title: 'Importancia de variables en Random Forest',
    hAxis: {title: 'Bandas'},
    vAxis: {title: 'Importancia'},
    // legend: {position: 'none'},
    // colors: ['#1f77b4']
  });
print(chart);

var test =finalImage.select(bands).sampleRegions
      ({
      collection: table,
      properties: ['b1'],
      scale: 10,
      tileScale: 16
      });

var numTreesList = ee.List.sequence(5, 150, 5);

var accuracies = numTreesList.map(function(numTrees)
    {
      var classifier = ee.Classifier.smileRandomForest(numTrees)
          .train(
            {
              features: training ,
              classProperty: 'b1',
              inputProperties: finalImage.select(bands).bandNames()
            }
      );

  return test
    .classify(classifier)
    .errorMatrix('b1', 'classification')
    .accuracy();
});

var chart = ui.Chart.array.values({
  array: ee.Array(accuracies),
  axis: 0,
  xLabels: numTreesList
  }).setOptions({
      title: 'Numero óptimo de Arboles decisión (numberOfTrees)',
      vAxis: {title: 'Validation Accuracy'},
      hAxis: {title: 'Number of Tress', gridlines: {count: 15}}
  });
print(chart)

var numTreesList = ee.List.sequence(5, 150, 5);
var bagFractionList = ee.List.sequence(0.1, 0.9, 0.1);

var accuracies = numTreesList.map(function(numTrees) {
  return bagFractionList.map(function(bagFraction) {
    var classifier = ee.Classifier.smileRandomForest({
      numberOfTrees: numTrees,
      bagFraction: bagFraction
    })
    .train(
      {
        features: table,
        classProperty: 'b1',
        inputProperties: finalImage.select(bands).bandNames()
      }
    );


    var accuracy = test
      .classify(classifier)
      .errorMatrix('b1', 'classification')
      .accuracy();
    return ee.Feature(null, {'accuracy': accuracy,
      'numberOfTrees': numTrees,
      'bagFraction': bagFraction})
  })
}).flatten()
var resultFc = ee.FeatureCollection(accuracies)
  
                        
// Define el umbral del filtro espacial para limpiar parches pequeños
var minConnectPixel = 4         /*umbral de min - max de pixeles */
var eightConnected = true

var patchsize = classified.unmask().connectedPixelCount(
        {
        'maxSize': 100, 
        'eightConnected': eightConnected 
        } 
      ); 

// Suaviza usando la moda local (valor más frecuente en vecindad).
var moda = classified.unmask().focal_mode
    (
      {
       'radius': 1,
       'kernelType': 'square',
       'units': 'pixels',
       'iterations': 1,
      }
    ).updateMask(patchsize.lte(minConnectPixel));

// Clasificación suavizada, conservando parches grandes y corrigiendo los pequeños.
var class_out =  classified.blend(moda);

// Muestra en el mapa la clasificacion con el filtro espacial.
Map.addLayer(class_out.reproject('EPSG:4326', null, 10), {min: 1, max: 2, palette: ['black', 'white']}, 'spatialFilter', false);

// Exporta al cloud asset
Export.image.toAsset(
      {
        'image': class_out.clip(carta).toInt8(),
        'description': fileName,
        'assetId':  'projects/ee-monitoreo2024/assets/' + fileName,
        'scale': 10,
        'crs': 'EPSG:4326',
        'region':  carta.geometry().bounds(),
        'maxPixels': 1e13,
        'pyramidingPolicy': 'mode',
      }
);                        

//Exporta al drive 
Export.image.toDrive(
      {
        image: class_out.clip(carta).toInt8(),
        description: fileName,
        scale: 10,
        crs: 'EPSG:4326',
        folder: 'GEE_Exports',
        maxPixels: 1e13,
        region: carta
    });

/*
	Joseph Sarabia
	CAP 6721
	Homework 5
*/

var resolution = 0;

 var sceneLoc = "Cornell_box_model.json";
// var type = 2;

//var sceneLoc = "sampleMesh.json";
var type = 0;

var radius = 0;
var far = 10;

function main(){
	 
	var maxint = 2147483647;

	var cl = WebCL.createContext ();
	var device = cl.getInfo(WebCL.CONTEXT_DEVICES)[0];
	var cmdQueue = cl.createCommandQueue (device, 0);
	var programSrc = loadKernel("raytrace");
	var program = cl.createProgram(programSrc);
	try {
		program.build ([device], "");
	} catch(e) {
		alert ("Failed to build WebCL program. Error "
		   + program.getBuildInfo (device, WebCL.PROGRAM_BUILD_STATUS)
		   + ":  " + program.getBuildInfo (device, WebCL.PROGRAM_BUILD_LOG));
		throw e;
	}
	var kernelName = "raytrace";
	try {
		kernel = program.createKernel (kernelName);
	} catch(e){
		alert("No kernel with name:"+ kernelName+" is found.");
		throw e;
	}
	var scene = new Scene(sceneLoc);
	var canvas = document.getElementById("canvas");
	var width=canvas.width, height=canvas.height;
	var canvasContext=canvas.getContext("2d");
	var canvasContent = canvasContext.createImageData(width,height);
	var nPixels = width*height;
	var nChannels = 4;
	var pixelBufferSize = nChannels*nPixels;
	var pixelBuffer = cl.createBuffer(WebCL.MEM_WRITE_ONLY,pixelBufferSize);
	var cameraBufferSize = 40;
	var cameraBuffer = cl.createBuffer(WebCL.MEM_WRITE_ONLY, cameraBufferSize);
	// [eye,at,up,fov]
	var cameraBufferData = new Float32Array([0,0,1,0,0,0,0,1,0,90]);
	var cameraObj = scene.getViewSpec(0);
	if (cameraObj)
	{
		cameraBufferData[0] = cameraObj.eye[0];
		cameraBufferData[1] = cameraObj.eye[1];
		cameraBufferData[2] = cameraObj.eye[2];
		cameraBufferData[3] = cameraObj.at[0];
		cameraBufferData[4] = cameraObj.at[1];
		cameraBufferData[5] = cameraObj.at[2];
		cameraBufferData[6] = cameraObj.up[0];
		cameraBufferData[7] = cameraObj.up[1];
		cameraBufferData[8] = cameraObj.up[2];
		cameraBufferData[9] = cameraObj.fov;
	}

	var triangleBufferSize = scene.getTriangleBufferSize();
	var triangleBuffer = cl.createBuffer(WebCL.MEM_WRITE_ONLY,(triangleBufferSize)?triangleBufferSize:1);
	var nTriangles = scene.getNtriangles();

	var bounds = scene.getBounds();
	boundsAABB = getAABBbounds(bounds);
	var sceneData = scene.getTriangleBufferData();
	var AABBresolution = [resolution,resolution,resolution];
	if(resolution == 0)
		AABBresolution = getResolution(bounds, nTriangles);
	var trianglePartition = partitionTriangles(boundsAABB, sceneData, AABBresolution, nTriangles);
	console.log(trianglePartition[0]);
	console.log("grid resolution is " +AABBresolution);
	console.log("bounds are ");
	console.log(boundsAABB);
	console.log("object count is " +nTriangles);

	var gridCells = [];
	var gridCellsIndex = [];
	 
	var count = 0;
	for(var i = 0; i < trianglePartition.length;i++){
		gridCellsIndex.push(count);
		var tcells = trianglePartition[i].getPrimitives();
		for(var j = 0; j < tcells.length; j++){
			gridCells.push(tcells[j]);
		}
		count += tcells.length;
	}



	var gridBufferSize = gridCells.length * 4;
	var gridBuffer = cl.createBuffer(WebCL.MEM_WRITE_ONLY, gridBufferSize);
	var nGridBuffer = gridCells.length;

	var gridIndexBufferSize = gridCellsIndex.length * 4;
	var gridIndexBuffer = cl.createBuffer(WebCL.MEM_WRITE_ONLY, gridIndexBufferSize);
	var nGridIndexBuffer = gridCellsIndex.length;
	
	var boundsData = new Float32Array([boundsAABB.min[0],boundsAABB.min[1],boundsAABB.min[2],boundsAABB.max[0],boundsAABB.max[1],boundsAABB.max[2]]);
	console.log("boundsData is ");
	console.log(boundsData);

	//var boundsData = new Float32Array([-4,-4,-4,4,4,4]);
	var boundsBuffer = cl.createBuffer(WebCL.MEM_WRITE_ONLY, 24);

	var nMaterials = scene.getNmaterials();
	var materialBufferSize = scene.getMaterialBufferSize();
	var materialBuffer = cl.createBuffer(WebCL.MEM_WRITE_ONLY, (materialBufferSize)?materialBufferSize:40);




	 kernel.setArg(0, pixelBuffer);
	 kernel.setArg(1, cameraBuffer);
	
	 kernel.setArg(2, triangleBuffer);
	 kernel.setArg(3, new Int32Array([nTriangles]));

	 kernel.setArg(4, new Int32Array([width]));
	 kernel.setArg(5, new Int32Array([height]));

	 kernel.setArg(6, new Int32Array([nMaterials]));
	 kernel.setArg(7, materialBuffer);

 
	kernel.setArg(8, gridBuffer);
	kernel.setArg(9, gridIndexBuffer);
	kernel.setArg(10, new Int32Array([nGridBuffer]));
	kernel.setArg(11, new Int32Array([nGridIndexBuffer]));
	kernel.setArg(12, boundsBuffer);
	kernel.setArg(13, new Int32Array([AABBresolution[0]]));
	kernel.setArg(14, new Int32Array([type]));
	console.log("nCells is "+ nGridIndexBuffer);


	var dim = 2;
	var maxWorkElements = kernel.getWorkGroupInfo(device,webCL.KERNEL_WORK_GROUP_SIZE);// WorkElements in ComputeUnit
	var xSize = Math.floor(Math.sqrt(maxWorkElements));
	var ySize = Math.floor(maxWorkElements/xSize);
	var localWS = [xSize, ySize];
	var globalWS = [Math.ceil(width/xSize)*xSize, Math.ceil(height/ySize)*ySize];

	cmdQueue.enqueueWriteBuffer(triangleBuffer, false, 0, triangleBufferSize, scene.getTriangleBufferData());
	cmdQueue.enqueueWriteBuffer(materialBuffer, false, 0, materialBufferSize, scene.getMaterialBufferData());
	cmdQueue.enqueueWriteBuffer(boundsBuffer, false, 0, 24, boundsData);
	cmdQueue.enqueueWriteBuffer(gridBuffer, false, 0, gridBufferSize, new Int32Array(gridCells));
	cmdQueue.enqueueWriteBuffer(gridIndexBuffer, false, 0, gridIndexBufferSize, new Int32Array(gridCellsIndex));
	
	console.log("Cells Structure Array is " + gridCells);
	console.log("Cells index Array is " + gridCellsIndex);
	console.log("size of cell buffer is " + nGridBuffer);

	cmdQueue.enqueueWriteBuffer(cameraBuffer, false, 0, cameraBufferSize, cameraBufferData);
	cmdQueue.enqueueNDRangeKernel(kernel,globalWS.length,null,globalWS,localWS);
	// Must be done by pushing a Read request to the command queue
	cmdQueue.enqueueReadBuffer(pixelBuffer,false,0,pixelBufferSize,canvasContent.data);
	cmdQueue.finish();
	canvasContext.putImageData(canvasContent,0,0);
	pixelBuffer.release();
	cameraBuffer.release();
	cmdQueue.release();
	kernel.release();
	program.release();
	cl.releaseAll();
	cl.release();
}

function loadKernel(id){
  var kernelElement = document.getElementById(id);
  console.log(document.getElementById(id));
  var kernelSource = kernelElement.text;
  if (kernelElement.src != "") {
      var mHttpReq = new XMLHttpRequest();
      mHttpReq.open("GET", kernelElement.src, false);
      mHttpReq.send(null);
      kernelSource = mHttpReq.responseText;
  } 
  return kernelSource;
}

function switchstuff(){
	if(sceneLoc == "Cornell_box_model.json"){
		type = 1;
		sceneLoc = "sampleMesh2.json";
	}
	else {
		sceneLoc = "Cornell_box_model.json";
		type = 0;
	}
	main();
}


function partitionTriangles(bounds, triangleData, resolution, nTriangles){

	var xSizeAABB = bounds.max[0] - bounds.min[0];
	var ySizeAABB = bounds.max[1] - bounds.min[1];
	var zSizeAABB = bounds.max[2] - bounds.min[2];

	var cellIndex = [];
	var trianglesIndex = [];
	trianglesIndex.push(0);
	for(var i = 0; i < nTriangles; i++){
		
		//get coordinates of min cell and max cell
		var minx = Math.min(triangleData[(19*i)+0], triangleData[(19*i)+3], triangleData[(19*i)+6]);
		var miny = Math.min(triangleData[(19*i)+1], triangleData[(19*i)+4], triangleData[(19*i)+7]);
		var minz = Math.min(triangleData[(19*i)+2], triangleData[(19*i)+5], triangleData[(19*i)+8]);

		var maxx = Math.max(triangleData[(19*i)+0], triangleData[(19*i)+3], triangleData[(19*i)+6]);
		var maxy = Math.max(triangleData[(19*i)+1], triangleData[(19*i)+4], triangleData[(19*i)+7]);
		var maxz = Math.max(triangleData[(19*i)+2], triangleData[(19*i)+5], triangleData[(19*i)+8]);

		var tmin = [minx,miny,minz];

		var tmax = [maxx,maxy,maxz];



		var minindex = [clamp(Math.floor((tmin[0]-bounds.min[0])/(xSizeAABB/resolution[0])), 0, resolution[0]-1),
					clamp(Math.floor((tmin[1]-bounds.min[1])/(ySizeAABB/resolution[1])), 0, resolution[1]-1),
					clamp(Math.floor((tmin[2]-bounds.min[2])/(zSizeAABB/resolution[2])), 0, resolution[2]-1)];
		console.log("tmin is " + tmin);
		console.log("minindex is " +minindex);

		var maxindex = [clamp(Math.floor((tmax[0]-bounds.min[0])/(xSizeAABB/resolution[0])), 0, resolution[0]-1),
					clamp(Math.floor((tmax[1]-bounds.min[1])/(ySizeAABB/resolution[1])), 0, resolution[1]-1),
					clamp(Math.floor((tmax[2]-bounds.min[2])/(zSizeAABB/resolution[2])), 0, resolution[2]-1)];

		console.log("tmax is " + tmax);
		console.log("maxindex is " +maxindex);


		for(var x = minindex[0]; x <= maxindex[0]; x++){
			for(var y = minindex[1]; y<= maxindex[1]; y++){
				for(var z = minindex[2]; z <= maxindex[2]; z++){
					cellIndex.push(x); cellIndex.push(y); cellIndex.push(z);
					console.log("adding "+ x + "," +y + "," +z);
				}
			}
		}
		
		
		trianglesIndex.push(cellIndex.length);

	}
	trianglesIndex.pop();
	console.log("cell index is " + cellIndex);
	console.log("trianglesIndex is " +trianglesIndex);
	var gridStructure = []
	//generate grid structure
	//loop through all the cells
	for(var x=0;x<resolution[0];x++)
		for(var y=0;y<resolution[1];y++)
			for(var z=0;z<resolution[2];z++){
				gridStructure.push(new cell(x, y, z));
			}

	//populate grid
	var updated = false; var cindex=0;
	//loop through all the cells
	for(var i=0;i<gridStructure.length;i++){
		var index=0; 
		//loop through the entire cell index array
		for(var j =0; j<cellIndex.length;j+=3){
			//keep track of which sphere you're on
			if(index+1 < trianglesIndex.length && j == (trianglesIndex[index+1]))
				index++;

			//push that sphere if the indices match
			if(cellIndex[j] == gridStructure[i].getX() && cellIndex[j+1] == gridStructure[i].getY() && cellIndex[j+2] == gridStructure[i].getZ()){
				gridStructure[i].addPrimitive(index);
				console.log("pushing " + index + " to cell");
			}
		}

	}
	console.log("Logging grid structure:");
	console.log(gridStructure);
	console.log("Grid structure logged");

	console.log("Printing cell contents:");
	for(var x=0;x<gridStructure.length;x++){
				console.log("contents of cell #:" +x);
				console.log(gridStructure[x].getPrimitives());
				//console.log();
			}
	
	return gridStructure;

}

function clamp(number, min, max){
	if (number < min)
		return min;
	else if (number > max)
		return max;
	else 
		return number;
}

function getResolution(bounds, numObjects){
	var x = bounds.max[0] - bounds.min[0];
	var y = bounds.max[1] - bounds.min[1];
	var z = bounds.max[2] - bounds.min[2];
	console.log("xsize is "+x+" ysize is "+y+" zsizeis "+z);
	var maxsize = (Math.max(Math.max(x,y),z));
	//var boxVolume = x*y*z;
	var a = (maxsize*3) / (Math.pow(numObjects, 1/3.0));
	console.log("a is "+a);
	var resolution = [];
	resolution[0] = clamp(Math.round(maxsize/a),1,64);
	resolution[1] = clamp(Math.round(maxsize/a),1,64);
	resolution[2] = clamp(Math.round(maxsize/a),1,64);
	return resolution;
}

function getAABBbounds(bounds){
	var x = bounds.max[0] - bounds.min[0];
	var y = bounds.max[1] - bounds.min[1];
	var z = bounds.max[2] - bounds.min[2];

	var axis = (Math.max(Math.max(x,y),z));

	return {min:bounds.min, max:[bounds.min[0]+axis, bounds.min[1]+axis, bounds.min[2]+axis]};
}


//cell object, stores its x y z position and indices to spheres it contains, 
//can return spheres index (returns -1 index if cell is empty),
//adds spheres, gets position
function cell(x,y,z)
{
	this.x = x;
	this.y = y;
	this.z = z;
	var primitives = [];

	this.getCellPosition=function(){return [x,y,z];};
	this.getX=function(){return x;};
	this.getY=function(){return y;};
	this.getZ=function(){return z;};
   	this.addPrimitive=function(a){primitives.push(a);};
    this.getPrimitives=function(){
    	if(primitives.length > 0)
    	return primitives;
    	else return [-1];
    };

}
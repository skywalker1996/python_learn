
//thread id 为 row, column
int startid = 0;
int endid = 0;

if(row<MATRIXSIZE && column<MATRIXSIZE){
	startid = column;
	endid = column + Arr_rows - 1;
	if(row>=startid && row<=endid){
		A[row*Width_A + column] = Arr[row - startid];
	}else{
		A[row*Width_A + column] = 0.0;
	}
}
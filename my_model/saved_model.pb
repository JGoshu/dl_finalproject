ď2
˙â
D
AddV2
x"T
y"T
z"T"
Ttype:
2	
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
l
BatchMatMulV2
x"T
y"T
output"T"
Ttype:
2		"
adj_xbool( "
adj_ybool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
8
Const
output"dtype"
valuetensor"
dtypetype
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
ű
FusedBatchNormV3
x"T

scale"U
offset"U	
mean"U
variance"U
y"T

batch_mean"U
batch_variance"U
reserve_space_1"U
reserve_space_2"U
reserve_space_3"U"
Ttype:
2"
Utype:
2"
epsilonfloat%ˇŃ8"&
exponential_avg_factorfloat%  ?";
data_formatstringNHWC:
NHWCNCHWNDHWCNCDHW"
is_trainingbool(
.
Identity

input"T
output"T"	
Ttype
\
	LeakyRelu
features"T
activations"T"
alphafloat%ÍĚL>"
Ttype0:
2
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	

Max

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	

MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( 
?
Mul
x"T
y"T
z"T"
Ttype:
2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
f
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx" 
Tidxtype0:
2
	
@
ReadVariableOp
resource
value"dtype"
dtypetype
@
RealDiv
x"T
y"T
z"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
Ľ
ResourceGather
resource
indices"Tindices
output"dtype"

batch_dimsint "
validate_indicesbool("
dtypetype"
Tindicestype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
9
Softmax
logits"T
softmax"T"
Ttype:
2
-
Sqrt
x"T
y"T"
Ttype:

2
Á
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ¨
@
StaticRegexFullMatch	
input

output
"
patternstring
÷
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.11.02unknown8Íś.
p
dense_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_4/bias
i
 dense_4/bias/Read/ReadVariableOpReadVariableOpdense_4/bias*
_output_shapes
:*
dtype0
y
dense_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Ź*
shared_namedense_4/kernel
r
"dense_4/kernel/Read/ReadVariableOpReadVariableOpdense_4/kernel*
_output_shapes
:	Ź*
dtype0

layer_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ź*+
shared_namelayer_normalization_1/beta

.layer_normalization_1/beta/Read/ReadVariableOpReadVariableOplayer_normalization_1/beta*
_output_shapes	
:Ź*
dtype0

layer_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ź*,
shared_namelayer_normalization_1/gamma

/layer_normalization_1/gamma/Read/ReadVariableOpReadVariableOplayer_normalization_1/gamma*
_output_shapes	
:Ź*
dtype0
n
VariableVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ŹŹ*
shared_name
Variable
g
Variable/Read/ReadVariableOpReadVariableOpVariable* 
_output_shapes
:
ŹŹ*
dtype0
r

Variable_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:
ŹŹ*
shared_name
Variable_1
k
Variable_1/Read/ReadVariableOpReadVariableOp
Variable_1* 
_output_shapes
:
ŹŹ*
dtype0
r

Variable_2VarHandleOp*
_output_shapes
: *
dtype0*
shape:
ŹŹ*
shared_name
Variable_2
k
Variable_2/Read/ReadVariableOpReadVariableOp
Variable_2* 
_output_shapes
:
ŹŹ*
dtype0
r

Variable_3VarHandleOp*
_output_shapes
: *
dtype0*
shape:
ŹŹ*
shared_name
Variable_3
k
Variable_3/Read/ReadVariableOpReadVariableOp
Variable_3* 
_output_shapes
:
ŹŹ*
dtype0
r

Variable_4VarHandleOp*
_output_shapes
: *
dtype0*
shape:
ŹŹ*
shared_name
Variable_4
k
Variable_4/Read/ReadVariableOpReadVariableOp
Variable_4* 
_output_shapes
:
ŹŹ*
dtype0
r

Variable_5VarHandleOp*
_output_shapes
: *
dtype0*
shape:
ŹŹ*
shared_name
Variable_5
k
Variable_5/Read/ReadVariableOpReadVariableOp
Variable_5* 
_output_shapes
:
ŹŹ*
dtype0
q
dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ź*
shared_namedense_3/bias
j
 dense_3/bias/Read/ReadVariableOpReadVariableOpdense_3/bias*
_output_shapes	
:Ź*
dtype0
z
dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ŹŹ*
shared_namedense_3/kernel
s
"dense_3/kernel/Read/ReadVariableOpReadVariableOpdense_3/kernel* 
_output_shapes
:
ŹŹ*
dtype0
q
dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ź*
shared_namedense_2/bias
j
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes	
:Ź*
dtype0
z
dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ŹŹ*
shared_namedense_2/kernel
s
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel* 
_output_shapes
:
ŹŹ*
dtype0

layer_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ź*)
shared_namelayer_normalization/beta

,layer_normalization/beta/Read/ReadVariableOpReadVariableOplayer_normalization/beta*
_output_shapes	
:Ź*
dtype0

layer_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ź**
shared_namelayer_normalization/gamma

-layer_normalization/gamma/Read/ReadVariableOpReadVariableOplayer_normalization/gamma*
_output_shapes	
:Ź*
dtype0
r

Variable_6VarHandleOp*
_output_shapes
: *
dtype0*
shape:
ŹŹ*
shared_name
Variable_6
k
Variable_6/Read/ReadVariableOpReadVariableOp
Variable_6* 
_output_shapes
:
ŹŹ*
dtype0
r

Variable_7VarHandleOp*
_output_shapes
: *
dtype0*
shape:
ŹŹ*
shared_name
Variable_7
k
Variable_7/Read/ReadVariableOpReadVariableOp
Variable_7* 
_output_shapes
:
ŹŹ*
dtype0
r

Variable_8VarHandleOp*
_output_shapes
: *
dtype0*
shape:
ŹŹ*
shared_name
Variable_8
k
Variable_8/Read/ReadVariableOpReadVariableOp
Variable_8* 
_output_shapes
:
ŹŹ*
dtype0
r

Variable_9VarHandleOp*
_output_shapes
: *
dtype0*
shape:
ŹŹ*
shared_name
Variable_9
k
Variable_9/Read/ReadVariableOpReadVariableOp
Variable_9* 
_output_shapes
:
ŹŹ*
dtype0
t
Variable_10VarHandleOp*
_output_shapes
: *
dtype0*
shape:
ŹŹ*
shared_nameVariable_10
m
Variable_10/Read/ReadVariableOpReadVariableOpVariable_10* 
_output_shapes
:
ŹŹ*
dtype0
t
Variable_11VarHandleOp*
_output_shapes
: *
dtype0*
shape:
ŹŹ*
shared_nameVariable_11
m
Variable_11/Read/ReadVariableOpReadVariableOpVariable_11* 
_output_shapes
:
ŹŹ*
dtype0
q
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ź*
shared_namedense_1/bias
j
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes	
:Ź*
dtype0
z
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ŹŹ*
shared_namedense_1/kernel
s
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel* 
_output_shapes
:
ŹŹ*
dtype0
m

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ź*
shared_name
dense/bias
f
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes	
:Ź*
dtype0
v
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ŹŹ*
shared_namedense/kernel
o
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel* 
_output_shapes
:
ŹŹ*
dtype0
ó
Kemotion_detection_model/token_and_position_embedding/embedding_1/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	dŹ*\
shared_nameMKemotion_detection_model/token_and_position_embedding/embedding_1/embeddings
ě
_emotion_detection_model/token_and_position_embedding/embedding_1/embeddings/Read/ReadVariableOpReadVariableOpKemotion_detection_model/token_and_position_embedding/embedding_1/embeddings*
_output_shapes
:	dŹ*
dtype0
đ
Iemotion_detection_model/token_and_position_embedding/embedding/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:
XŹ*Z
shared_nameKIemotion_detection_model/token_and_position_embedding/embedding/embeddings
é
]emotion_detection_model/token_and_position_embedding/embedding/embeddings/Read/ReadVariableOpReadVariableOpIemotion_detection_model/token_and_position_embedding/embedding/embeddings* 
_output_shapes
:
XŹ*
dtype0
r
serving_default_input_1Placeholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙
ŕ
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1Kemotion_detection_model/token_and_position_embedding/embedding_1/embeddingsIemotion_detection_model/token_and_position_embedding/embedding/embeddingsVariable_11Variable_10
Variable_9layer_normalization/gammalayer_normalization/betadense/kernel
dense/biasdense_1/kerneldense_1/bias
Variable_5
Variable_4
Variable_3layer_normalization_1/gammalayer_normalization_1/beta
Variable_2
Variable_1Variabledense_2/kerneldense_2/biasdense_3/kerneldense_3/biasdense_4/kerneldense_4/bias*%
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*;
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *.
f)R'
%__inference_signature_wrapper_3833772

NoOpNoOp
Ź
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ć
valueŰB× BĎ

	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
embedding_layer
	enc_positional_embedding

enc_transformerblock
word2idx
accuracy_list
dec_transformer_block
dec_positional_embedding
dec_pooling
dec_classifier
dropout

signatures*
Ú
0
1
2
3
4
5
6
7
8
9
10
11
12
 13
!14
"15
#16
$17
%18
&19
'20
(21
)22
*23
+24
,25
-26
.27*
Ú
0
1
2
3
4
5
6
7
8
9
10
11
12
 13
!14
"15
#16
$17
%18
&19
'20
(21
)22
*23
+24
,25
-26
.27*
* 
°
/non_trainable_variables

0layers
1metrics
2layer_regularization_losses
3layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
4trace_0
5trace_1
6trace_2
7trace_3* 
6
8trace_0
9trace_1
:trace_2
;trace_3* 
* 
 
<	variables
=trainable_variables
>regularization_losses
?	keras_api
@__call__
*A&call_and_return_all_conditional_losses

embeddings*
Ź
B	variables
Ctrainable_variables
Dregularization_losses
E	keras_api
F__call__
*G&call_and_return_all_conditional_losses
	token_emb
Hpos_emb*
ŕ
I	variables
Jtrainable_variables
Kregularization_losses
L	keras_api
M__call__
*N&call_and_return_all_conditional_losses
Off_layer
P
self_atten
Qself_context_atten
R
layer_norm
Scall*
* 
* 
ŕ
T	variables
Utrainable_variables
Vregularization_losses
W	keras_api
X__call__
*Y&call_and_return_all_conditional_losses
Zff_layer
[
self_atten
\self_context_atten
]
layer_norm
^call*
/
_	keras_api
	token_emb
`pos_emb*

a	variables
btrainable_variables
cregularization_losses
d	keras_api
e__call__
*f&call_and_return_all_conditional_losses* 
ˇ
glayer_with_weights-0
glayer-0
h	variables
itrainable_variables
jregularization_losses
k	keras_api
l__call__
*m&call_and_return_all_conditional_losses*
Ľ
n	variables
otrainable_variables
pregularization_losses
q	keras_api
r__call__
*s&call_and_return_all_conditional_losses
t_random_generator* 

userving_default* 

VARIABLE_VALUEIemotion_detection_model/token_and_position_embedding/embedding/embeddings&variables/0/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEKemotion_detection_model/token_and_position_embedding/embedding_1/embeddings&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEdense/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
JD
VARIABLE_VALUE
dense/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEdense_1/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEdense_1/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
KE
VARIABLE_VALUEVariable_11&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
KE
VARIABLE_VALUEVariable_10&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
JD
VARIABLE_VALUE
Variable_9&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
JD
VARIABLE_VALUE
Variable_8&variables/9/.ATTRIBUTES/VARIABLE_VALUE*
KE
VARIABLE_VALUE
Variable_7'variables/10/.ATTRIBUTES/VARIABLE_VALUE*
KE
VARIABLE_VALUE
Variable_6'variables/11/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElayer_normalization/gamma'variables/12/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUElayer_normalization/beta'variables/13/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEdense_2/kernel'variables/14/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEdense_2/bias'variables/15/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEdense_3/kernel'variables/16/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEdense_3/bias'variables/17/.ATTRIBUTES/VARIABLE_VALUE*
KE
VARIABLE_VALUE
Variable_5'variables/18/.ATTRIBUTES/VARIABLE_VALUE*
KE
VARIABLE_VALUE
Variable_4'variables/19/.ATTRIBUTES/VARIABLE_VALUE*
KE
VARIABLE_VALUE
Variable_3'variables/20/.ATTRIBUTES/VARIABLE_VALUE*
KE
VARIABLE_VALUE
Variable_2'variables/21/.ATTRIBUTES/VARIABLE_VALUE*
KE
VARIABLE_VALUE
Variable_1'variables/22/.ATTRIBUTES/VARIABLE_VALUE*
IC
VARIABLE_VALUEVariable'variables/23/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUElayer_normalization_1/gamma'variables/24/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUElayer_normalization_1/beta'variables/25/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEdense_4/kernel'variables/26/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEdense_4/bias'variables/27/.ATTRIBUTES/VARIABLE_VALUE*
* 
<
0
	1

2
3
4
5
6
7*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

0*

0*
* 

vnon_trainable_variables

wlayers
xmetrics
ylayer_regularization_losses
zlayer_metrics
<	variables
=trainable_variables
>regularization_losses
@__call__
*A&call_and_return_all_conditional_losses
&A"call_and_return_conditional_losses*

{trace_0* 

|trace_0* 

0
1*

0
1*
* 

}non_trainable_variables

~layers
metrics
 layer_regularization_losses
layer_metrics
B	variables
Ctrainable_variables
Dregularization_losses
F__call__
*G&call_and_return_all_conditional_losses
&G"call_and_return_conditional_losses*

trace_0* 

trace_0* 
Ś
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses

embeddings*
Z
0
1
2
3
4
5
6
7
8
9
10
 11*
Z
0
1
2
3
4
5
6
7
8
9
10
 11*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
I	variables
Jtrainable_variables
Kregularization_losses
M__call__
*N&call_and_return_all_conditional_losses
&N"call_and_return_conditional_losses*

trace_0
trace_1* 

trace_0
trace_1* 
č
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses*
Í
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+ &call_and_return_all_conditional_losses
K
V
Q
Ąattention_matrix
	˘call*
K
Ł	keras_api
K
V
Q
¤attention_matrix
	Ľcall*
ś
Ś	variables
§trainable_variables
¨regularization_losses
Š	keras_api
Ş__call__
+Ť&call_and_return_all_conditional_losses
	Źaxis
	gamma
 beta*

­trace_0* 
Z
!0
"1
#2
$3
%4
&5
'6
(7
)8
*9
+10
,11*
Z
!0
"1
#2
$3
%4
&5
'6
(7
)8
*9
+10
,11*
* 

Žnon_trainable_variables
Żlayers
°metrics
 ąlayer_regularization_losses
˛layer_metrics
T	variables
Utrainable_variables
Vregularization_losses
X__call__
*Y&call_and_return_all_conditional_losses
&Y"call_and_return_conditional_losses*

łtrace_0
´trace_1* 

ľtrace_0
śtrace_1* 
č
ˇlayer_with_weights-0
ˇlayer-0
¸layer_with_weights-1
¸layer-1
š	variables
ştrainable_variables
ťregularization_losses
ź	keras_api
˝__call__
+ž&call_and_return_all_conditional_losses*
Í
ż	variables
Ŕtrainable_variables
Áregularization_losses
Â	keras_api
Ă__call__
+Ä&call_and_return_all_conditional_losses
%K
&V
'Q
Ĺattention_matrix
	Ćcall*
Í
Ç	variables
Čtrainable_variables
Éregularization_losses
Ę	keras_api
Ë__call__
+Ě&call_and_return_all_conditional_losses
(K
)V
*Q
Íattention_matrix
	Îcall*
ś
Ď	variables
Đtrainable_variables
Ńregularization_losses
Ň	keras_api
Ó__call__
+Ô&call_and_return_all_conditional_losses
	Őaxis
	+gamma
,beta*

Ötrace_0* 
* 

×	keras_api* 
* 
* 
* 

Řnon_trainable_variables
Ůlayers
Úmetrics
 Űlayer_regularization_losses
Ülayer_metrics
a	variables
btrainable_variables
cregularization_losses
e__call__
*f&call_and_return_all_conditional_losses
&f"call_and_return_conditional_losses* 

Ýtrace_0* 

Ţtrace_0* 
Ź
ß	variables
ŕtrainable_variables
áregularization_losses
â	keras_api
ă__call__
+ä&call_and_return_all_conditional_losses

-kernel
.bias*

-0
.1*

-0
.1*
* 

ĺnon_trainable_variables
ćlayers
çmetrics
 člayer_regularization_losses
élayer_metrics
h	variables
itrainable_variables
jregularization_losses
l__call__
*m&call_and_return_all_conditional_losses
&m"call_and_return_conditional_losses*
:
ętrace_0
ëtrace_1
ětrace_2
ítrace_3* 
:
îtrace_0
ďtrace_1
đtrace_2
ńtrace_3* 
* 
* 
* 

ňnon_trainable_variables
ólayers
ômetrics
 őlayer_regularization_losses
ölayer_metrics
n	variables
otrainable_variables
pregularization_losses
r__call__
*s&call_and_return_all_conditional_losses
&s"call_and_return_conditional_losses* 

÷trace_0
řtrace_1* 

ůtrace_0
útrace_1* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

0
H1*
* 
* 
* 
* 
* 

0*

0*
* 

űnon_trainable_variables
ülayers
ýmetrics
 ţlayer_regularization_losses
˙layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*
* 
* 
* 
 
O0
P1
Q2
R3*
* 
* 
* 
* 
* 
* 
* 
Ź
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses

kernel
bias*
Ź
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses

kernel
bias*
 
0
1
2
3*
 
0
1
2
3*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*
:
trace_0
trace_1
trace_2
trace_3* 
:
trace_0
trace_1
trace_2
trace_3* 

0
1
2*

0
1
2*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+ &call_and_return_all_conditional_losses
' "call_and_return_conditional_losses*
* 
* 

	variables
trainable_variables
 regularization_losses
Ą	keras_api
˘__call__
+Ł&call_and_return_all_conditional_losses* 

¤trace_0* 
* 

Ľ	keras_api* 
* 

0
 1*

0
 1*
* 

Śnon_trainable_variables
§layers
¨metrics
 Šlayer_regularization_losses
Şlayer_metrics
Ś	variables
§trainable_variables
¨regularization_losses
Ş__call__
+Ť&call_and_return_all_conditional_losses
'Ť"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
 
Z0
[1
\2
]3*
* 
* 
* 
* 
* 
* 
* 
Ź
Ť	variables
Źtrainable_variables
­regularization_losses
Ž	keras_api
Ż__call__
+°&call_and_return_all_conditional_losses

!kernel
"bias*
Ź
ą	variables
˛trainable_variables
łregularization_losses
´	keras_api
ľ__call__
+ś&call_and_return_all_conditional_losses

#kernel
$bias*
 
!0
"1
#2
$3*
 
!0
"1
#2
$3*
* 

ˇnon_trainable_variables
¸layers
šmetrics
 şlayer_regularization_losses
ťlayer_metrics
š	variables
ştrainable_variables
ťregularization_losses
˝__call__
+ž&call_and_return_all_conditional_losses
'ž"call_and_return_conditional_losses*
:
źtrace_0
˝trace_1
žtrace_2
żtrace_3* 
:
Ŕtrace_0
Átrace_1
Âtrace_2
Ătrace_3* 

%0
&1
'2*

%0
&1
'2*
* 

Änon_trainable_variables
Ĺlayers
Ćmetrics
 Çlayer_regularization_losses
Člayer_metrics
ż	variables
Ŕtrainable_variables
Áregularization_losses
Ă__call__
+Ä&call_and_return_all_conditional_losses
'Ä"call_and_return_conditional_losses*
* 
* 

É	variables
Ętrainable_variables
Ëregularization_losses
Ě	keras_api
Í__call__
+Î&call_and_return_all_conditional_losses* 

Ďtrace_0* 

(0
)1
*2*

(0
)1
*2*
* 

Đnon_trainable_variables
Ńlayers
Ňmetrics
 Ólayer_regularization_losses
Ôlayer_metrics
Ç	variables
Čtrainable_variables
Éregularization_losses
Ë__call__
+Ě&call_and_return_all_conditional_losses
'Ě"call_and_return_conditional_losses*
* 
* 

Ő	variables
Ötrainable_variables
×regularization_losses
Ř	keras_api
Ů__call__
+Ú&call_and_return_all_conditional_losses* 

Űtrace_0* 

+0
,1*

+0
,1*
* 

Ünon_trainable_variables
Ýlayers
Ţmetrics
 ßlayer_regularization_losses
ŕlayer_metrics
Ď	variables
Đtrainable_variables
Ńregularization_losses
Ó__call__
+Ô&call_and_return_all_conditional_losses
'Ô"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

-0
.1*

-0
.1*
* 

ánon_trainable_variables
âlayers
ămetrics
 älayer_regularization_losses
ĺlayer_metrics
ß	variables
ŕtrainable_variables
áregularization_losses
ă__call__
+ä&call_and_return_all_conditional_losses
'ä"call_and_return_conditional_losses*

ćtrace_0* 

çtrace_0* 
* 

g0*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

0
1*

0
1*
* 

čnon_trainable_variables
élayers
ęmetrics
 ëlayer_regularization_losses
ělayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*

ítrace_0* 

îtrace_0* 

0
1*

0
1*
* 

ďnon_trainable_variables
đlayers
ńmetrics
 ňlayer_regularization_losses
ólayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*

ôtrace_0* 

őtrace_0* 
* 

0
1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 


Ą0* 
* 
* 
* 
* 
* 
* 

önon_trainable_variables
÷layers
řmetrics
 ůlayer_regularization_losses
úlayer_metrics
	variables
trainable_variables
 regularization_losses
˘__call__
+Ł&call_and_return_all_conditional_losses
'Ł"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

!0
"1*

!0
"1*
* 

űnon_trainable_variables
ülayers
ýmetrics
 ţlayer_regularization_losses
˙layer_metrics
Ť	variables
Źtrainable_variables
­regularization_losses
Ż__call__
+°&call_and_return_all_conditional_losses
'°"call_and_return_conditional_losses*

trace_0* 

trace_0* 

#0
$1*

#0
$1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
ą	variables
˛trainable_variables
łregularization_losses
ľ__call__
+ś&call_and_return_all_conditional_losses
'ś"call_and_return_conditional_losses*

trace_0* 

trace_0* 
* 

ˇ0
¸1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 


Ĺ0* 
* 
* 
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
É	variables
Ętrainable_variables
Ëregularization_losses
Í__call__
+Î&call_and_return_all_conditional_losses
'Î"call_and_return_conditional_losses* 
* 
* 
* 
* 


Í0* 
* 
* 
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
Ő	variables
Ötrainable_variables
×regularization_losses
Ů__call__
+Ú&call_and_return_all_conditional_losses
'Ú"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename]emotion_detection_model/token_and_position_embedding/embedding/embeddings/Read/ReadVariableOp_emotion_detection_model/token_and_position_embedding/embedding_1/embeddings/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOpVariable_11/Read/ReadVariableOpVariable_10/Read/ReadVariableOpVariable_9/Read/ReadVariableOpVariable_8/Read/ReadVariableOpVariable_7/Read/ReadVariableOpVariable_6/Read/ReadVariableOp-layer_normalization/gamma/Read/ReadVariableOp,layer_normalization/beta/Read/ReadVariableOp"dense_2/kernel/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOp"dense_3/kernel/Read/ReadVariableOp dense_3/bias/Read/ReadVariableOpVariable_5/Read/ReadVariableOpVariable_4/Read/ReadVariableOpVariable_3/Read/ReadVariableOpVariable_2/Read/ReadVariableOpVariable_1/Read/ReadVariableOpVariable/Read/ReadVariableOp/layer_normalization_1/gamma/Read/ReadVariableOp.layer_normalization_1/beta/Read/ReadVariableOp"dense_4/kernel/Read/ReadVariableOp dense_4/bias/Read/ReadVariableOpConst*)
Tin"
 2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *)
f$R"
 __inference__traced_save_3835404
Ű
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameIemotion_detection_model/token_and_position_embedding/embedding/embeddingsKemotion_detection_model/token_and_position_embedding/embedding_1/embeddingsdense/kernel
dense/biasdense_1/kerneldense_1/biasVariable_11Variable_10
Variable_9
Variable_8
Variable_7
Variable_6layer_normalization/gammalayer_normalization/betadense_2/kerneldense_2/biasdense_3/kerneldense_3/bias
Variable_5
Variable_4
Variable_3
Variable_2
Variable_1Variablelayer_normalization_1/gammalayer_normalization_1/betadense_4/kerneldense_4/bias*(
Tin!
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *,
f'R%
#__inference__traced_restore_3835498ĂŘ,
ďu
ň
#__inference__traced_restore_3835498
file_prefixn
Zassignvariableop_emotion_detection_model_token_and_position_embedding_embedding_embeddings:
XŹq
^assignvariableop_1_emotion_detection_model_token_and_position_embedding_embedding_1_embeddings:	dŹ3
assignvariableop_2_dense_kernel:
ŹŹ,
assignvariableop_3_dense_bias:	Ź5
!assignvariableop_4_dense_1_kernel:
ŹŹ.
assignvariableop_5_dense_1_bias:	Ź2
assignvariableop_6_variable_11:
ŹŹ2
assignvariableop_7_variable_10:
ŹŹ1
assignvariableop_8_variable_9:
ŹŹ1
assignvariableop_9_variable_8:
ŹŹ2
assignvariableop_10_variable_7:
ŹŹ2
assignvariableop_11_variable_6:
ŹŹ<
-assignvariableop_12_layer_normalization_gamma:	Ź;
,assignvariableop_13_layer_normalization_beta:	Ź6
"assignvariableop_14_dense_2_kernel:
ŹŹ/
 assignvariableop_15_dense_2_bias:	Ź6
"assignvariableop_16_dense_3_kernel:
ŹŹ/
 assignvariableop_17_dense_3_bias:	Ź2
assignvariableop_18_variable_5:
ŹŹ2
assignvariableop_19_variable_4:
ŹŹ2
assignvariableop_20_variable_3:
ŹŹ2
assignvariableop_21_variable_2:
ŹŹ2
assignvariableop_22_variable_1:
ŹŹ0
assignvariableop_23_variable:
ŹŹ>
/assignvariableop_24_layer_normalization_1_gamma:	Ź=
.assignvariableop_25_layer_normalization_1_beta:	Ź5
"assignvariableop_26_dense_4_kernel:	Ź.
 assignvariableop_27_dense_4_bias:
identity_29˘AssignVariableOp˘AssignVariableOp_1˘AssignVariableOp_10˘AssignVariableOp_11˘AssignVariableOp_12˘AssignVariableOp_13˘AssignVariableOp_14˘AssignVariableOp_15˘AssignVariableOp_16˘AssignVariableOp_17˘AssignVariableOp_18˘AssignVariableOp_19˘AssignVariableOp_2˘AssignVariableOp_20˘AssignVariableOp_21˘AssignVariableOp_22˘AssignVariableOp_23˘AssignVariableOp_24˘AssignVariableOp_25˘AssignVariableOp_26˘AssignVariableOp_27˘AssignVariableOp_3˘AssignVariableOp_4˘AssignVariableOp_5˘AssignVariableOp_6˘AssignVariableOp_7˘AssignVariableOp_8˘AssignVariableOp_9˙	
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Ľ	
value	B	B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHŞ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*M
valueDBBB B B B B B B B B B B B B B B B B B B B B B B B B B B B B °
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*
_output_shapesv
t:::::::::::::::::::::::::::::*+
dtypes!
2[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:í
AssignVariableOpAssignVariableOpZassignvariableop_emotion_detection_model_token_and_position_embedding_embedding_embeddingsIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:ő
AssignVariableOp_1AssignVariableOp^assignvariableop_1_emotion_detection_model_token_and_position_embedding_embedding_1_embeddingsIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:ś
AssignVariableOp_2AssignVariableOpassignvariableop_2_dense_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:´
AssignVariableOp_3AssignVariableOpassignvariableop_3_dense_biasIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:¸
AssignVariableOp_4AssignVariableOp!assignvariableop_4_dense_1_kernelIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:ś
AssignVariableOp_5AssignVariableOpassignvariableop_5_dense_1_biasIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:ľ
AssignVariableOp_6AssignVariableOpassignvariableop_6_variable_11Identity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:ľ
AssignVariableOp_7AssignVariableOpassignvariableop_7_variable_10Identity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:´
AssignVariableOp_8AssignVariableOpassignvariableop_8_variable_9Identity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:´
AssignVariableOp_9AssignVariableOpassignvariableop_9_variable_8Identity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:ˇ
AssignVariableOp_10AssignVariableOpassignvariableop_10_variable_7Identity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:ˇ
AssignVariableOp_11AssignVariableOpassignvariableop_11_variable_6Identity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:Ć
AssignVariableOp_12AssignVariableOp-assignvariableop_12_layer_normalization_gammaIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:Ĺ
AssignVariableOp_13AssignVariableOp,assignvariableop_13_layer_normalization_betaIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:ť
AssignVariableOp_14AssignVariableOp"assignvariableop_14_dense_2_kernelIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:š
AssignVariableOp_15AssignVariableOp assignvariableop_15_dense_2_biasIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:ť
AssignVariableOp_16AssignVariableOp"assignvariableop_16_dense_3_kernelIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:š
AssignVariableOp_17AssignVariableOp assignvariableop_17_dense_3_biasIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:ˇ
AssignVariableOp_18AssignVariableOpassignvariableop_18_variable_5Identity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:ˇ
AssignVariableOp_19AssignVariableOpassignvariableop_19_variable_4Identity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:ˇ
AssignVariableOp_20AssignVariableOpassignvariableop_20_variable_3Identity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:ˇ
AssignVariableOp_21AssignVariableOpassignvariableop_21_variable_2Identity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:ˇ
AssignVariableOp_22AssignVariableOpassignvariableop_22_variable_1Identity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:ľ
AssignVariableOp_23AssignVariableOpassignvariableop_23_variableIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:Č
AssignVariableOp_24AssignVariableOp/assignvariableop_24_layer_normalization_1_gammaIdentity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:Ç
AssignVariableOp_25AssignVariableOp.assignvariableop_25_layer_normalization_1_betaIdentity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:ť
AssignVariableOp_26AssignVariableOp"assignvariableop_26_dense_4_kernelIdentity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:š
AssignVariableOp_27AssignVariableOp assignvariableop_27_dense_4_biasIdentity_27:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 ˇ
Identity_28Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_29IdentityIdentity_28:output:0^NoOp_1*
T0*
_output_shapes
: ¤
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_29Identity_29:output:0*M
_input_shapes<
:: : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
Ç
b
D__inference_dropout_layer_call_and_return_conditional_losses_3834827

inputs

identity_1J
IdentityIdentityinputs*
T0*#
_output_shapes
:ddŹW

Identity_1IdentityIdentity:output:0*
T0*#
_output_shapes
:ddŹ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*"
_input_shapes
:ddŹ:K G
#
_output_shapes
:ddŹ
 
_user_specified_nameinputs
Ď
Ţ
.__inference_sequential_1_layer_call_fn_3832351
dense_2_input
unknown:
ŹŹ
	unknown_0:	Ź
	unknown_1:
ŹŹ
	unknown_2:	Ź
identity˘StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCalldense_2_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:˙˙˙˙˙˙˙˙˙dŹ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sequential_1_layer_call_and_return_conditional_losses_3832327t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙dŹ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :˙˙˙˙˙˙˙˙˙dŹ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
,
_output_shapes
:˙˙˙˙˙˙˙˙˙dŹ
'
_user_specified_namedense_2_input
ş
×
.__inference_sequential_1_layer_call_fn_3834992

inputs
unknown:
ŹŹ
	unknown_0:	Ź
	unknown_1:
ŹŹ
	unknown_2:	Ź
identity˘StatefulPartitionedCallý
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:˙˙˙˙˙˙˙˙˙dŹ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sequential_1_layer_call_and_return_conditional_losses_3832267t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙dŹ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :˙˙˙˙˙˙˙˙˙dŹ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:˙˙˙˙˙˙˙˙˙dŹ
 
_user_specified_nameinputs
ő
 
Y__inference_token_and_position_embedding_layer_call_and_return_conditional_losses_3834083
x7
$embedding_1_embedding_lookup_3834071:	dŹ6
"embedding_embedding_lookup_3834076:
XŹ
identity˘embedding/embedding_lookup˘embedding_1/embedding_lookupM
range/startConst*
_output_shapes
: *
dtype0*
value	B : M
range/limitConst*
_output_shapes
: *
dtype0*
value	B :dM
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :l
rangeRangerange/start:output:0range/limit:output:0range/delta:output:0*
_output_shapes
:dŰ
embedding_1/embedding_lookupResourceGather$embedding_1_embedding_lookup_3834071range:output:0*
Tindices0*7
_class-
+)loc:@embedding_1/embedding_lookup/3834071*
_output_shapes
:	dŹ*
dtype0ť
%embedding_1/embedding_lookup/IdentityIdentity%embedding_1/embedding_lookup:output:0*
T0*7
_class-
+)loc:@embedding_1/embedding_lookup/3834071*
_output_shapes
:	dŹ
'embedding_1/embedding_lookup/Identity_1Identity.embedding_1/embedding_lookup/Identity:output:0*
T0*
_output_shapes
:	dŹŃ
embedding/embedding_lookupResourceGather"embedding_embedding_lookup_3834076x*
Tindices0*5
_class+
)'loc:@embedding/embedding_lookup/3834076*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ź*
dtype0ž
#embedding/embedding_lookup/IdentityIdentity#embedding/embedding_lookup:output:0*
T0*5
_class+
)'loc:@embedding/embedding_lookup/3834076*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ź
%embedding/embedding_lookup/Identity_1Identity,embedding/embedding_lookup/Identity:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ź
addAddV2.embedding/embedding_lookup/Identity_1:output:00embedding_1/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:	dŹN
IdentityIdentityadd:z:0^NoOp*
T0*
_output_shapes
:	dŹ
NoOpNoOp^embedding/embedding_lookup^embedding_1/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:˙˙˙˙˙˙˙˙˙: : 28
embedding/embedding_lookupembedding/embedding_lookup2<
embedding_1/embedding_lookupembedding_1/embedding_lookup:F B
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_namex


c
D__inference_dropout_layer_call_and_return_conditional_losses_3833178

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @`
dropout/MulMulinputsdropout/Const:output:0*
T0*#
_output_shapes
:ddŹb
dropout/ShapeConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ,  
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*#
_output_shapes
:ddŹ*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?˘
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*#
_output_shapes
:ddŹT
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*#
_output_shapes
:ddŹ]
IdentityIdentitydropout/SelectV2:output:0*
T0*#
_output_shapes
:ddŹ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*"
_input_shapes
:ddŹ:K G
#
_output_shapes
:ddŹ
 
_user_specified_nameinputs
×

I__inference_sequential_2_layer_call_and_return_conditional_losses_3834801

inputs9
&dense_4_matmul_readvariableop_resource:	Ź5
'dense_4_biasadd_readvariableop_resource:
identity˘dense_4/BiasAdd/ReadVariableOp˘dense_4/MatMul/ReadVariableOp
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes
:	Ź*
dtype0y
dense_4/MatMulMatMulinputs%dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙f
dense_4/SigmoidSigmoiddense_4/BiasAdd:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙b
IdentityIdentitydense_4/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
NoOpNoOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:˙˙˙˙˙˙˙˙˙Ź: : 2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ź
 
_user_specified_nameinputs
˙-
Ů
T__inference_emotion_detection_model_layer_call_and_return_conditional_losses_3833715
input_17
$token_and_position_embedding_3833656:	dŹ8
$token_and_position_embedding_3833658:
XŹ-
transformer_block_3833661:
ŹŹ-
transformer_block_3833663:
ŹŹ-
transformer_block_3833665:
ŹŹ(
transformer_block_3833667:	Ź(
transformer_block_3833669:	Ź-
transformer_block_3833671:
ŹŹ(
transformer_block_3833673:	Ź-
transformer_block_3833675:
ŹŹ(
transformer_block_3833677:	Ź/
transformer_block_1_3833681:
ŹŹ/
transformer_block_1_3833683:
ŹŹ/
transformer_block_1_3833685:
ŹŹ*
transformer_block_1_3833687:	Ź*
transformer_block_1_3833689:	Ź/
transformer_block_1_3833691:
ŹŹ/
transformer_block_1_3833693:
ŹŹ/
transformer_block_1_3833695:
ŹŹ/
transformer_block_1_3833697:
ŹŹ*
transformer_block_1_3833699:	Ź/
transformer_block_1_3833701:
ŹŹ*
transformer_block_1_3833703:	Ź'
sequential_2_3833709:	Ź"
sequential_2_3833711:
identity˘dropout/StatefulPartitionedCall˘$sequential_2/StatefulPartitionedCall˘4token_and_position_embedding/StatefulPartitionedCall˘)transformer_block/StatefulPartitionedCall˘+transformer_block_1/StatefulPartitionedCallź
4token_and_position_embedding/StatefulPartitionedCallStatefulPartitionedCallinput_1$token_and_position_embedding_3833656$token_and_position_embedding_3833658*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	dŹ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *b
f]R[
Y__inference_token_and_position_embedding_layer_call_and_return_conditional_losses_3832522
)transformer_block/StatefulPartitionedCallStatefulPartitionedCall=token_and_position_embedding/StatefulPartitionedCall:output:0transformer_block_3833661transformer_block_3833663transformer_block_3833665transformer_block_3833667transformer_block_3833669transformer_block_3833671transformer_block_3833673transformer_block_3833675transformer_block_3833677*
Tin
2
*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:ddŹ*+
_read_only_resource_inputs
		*-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_transformer_block_layer_call_and_return_conditional_losses_3833325ď
dropout/StatefulPartitionedCallStatefulPartitionedCall2transformer_block/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:ddŹ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_layer_call_and_return_conditional_losses_3833178ł
+transformer_block_1/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0=token_and_position_embedding/StatefulPartitionedCall:output:0transformer_block_1_3833681transformer_block_1_3833683transformer_block_1_3833685transformer_block_1_3833687transformer_block_1_3833689transformer_block_1_3833691transformer_block_1_3833693transformer_block_1_3833695transformer_block_1_3833697transformer_block_1_3833699transformer_block_1_3833701transformer_block_1_3833703*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:ddŹ*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_transformer_block_1_layer_call_and_return_conditional_losses_3833131P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 

ExpandDims
ExpandDims4transformer_block_1/StatefulPartitionedCall:output:0ExpandDims/dim:output:0*
T0*'
_output_shapes
:ddŹÖ
$global_max_pooling2d/PartitionedCallPartitionedCallExpandDims:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	Ź* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Z
fURS
Q__inference_global_max_pooling2d_layer_call_and_return_conditional_losses_3832389Ą
$sequential_2/StatefulPartitionedCallStatefulPartitionedCall-global_max_pooling2d/PartitionedCall:output:0sequential_2_3833709sequential_2_3833711*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sequential_2_layer_call_and_return_conditional_losses_3832454s
IdentityIdentity-sequential_2/StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

: 
NoOpNoOp ^dropout/StatefulPartitionedCall%^sequential_2/StatefulPartitionedCall5^token_and_position_embedding/StatefulPartitionedCall*^transformer_block/StatefulPartitionedCall,^transformer_block_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:˙˙˙˙˙˙˙˙˙: : : : : : : : : : : : : : : : : : : : : : : : : 2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2L
$sequential_2/StatefulPartitionedCall$sequential_2/StatefulPartitionedCall2l
4token_and_position_embedding/StatefulPartitionedCall4token_and_position_embedding/StatefulPartitionedCall2V
)transformer_block/StatefulPartitionedCall)transformer_block/StatefulPartitionedCall2Z
+transformer_block_1/StatefulPartitionedCall+transformer_block_1/StatefulPartitionedCall:L H
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
!
_user_specified_name	input_1
ş
Ś
F__inference_embedding_layer_call_and_return_conditional_losses_3834055

inputs,
embedding_lookup_3834049:
XŹ
identity˘embedding_lookup¸
embedding_lookupResourceGatherembedding_lookup_3834049inputs*
Tindices0*+
_class!
loc:@embedding_lookup/3834049*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ź*
dtype0 
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*+
_class!
loc:@embedding_lookup/3834049*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ź~
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Źt
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ŹY
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:˙˙˙˙˙˙˙˙˙: 2$
embedding_lookupembedding_lookup:K G
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
ś
Ő
,__inference_sequential_layer_call_fn_3834852

inputs
unknown:
ŹŹ
	unknown_0:	Ź
	unknown_1:
ŹŹ
	unknown_2:	Ź
identity˘StatefulPartitionedCallű
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:˙˙˙˙˙˙˙˙˙dŹ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_sequential_layer_call_and_return_conditional_losses_3832074t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙dŹ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :˙˙˙˙˙˙˙˙˙dŹ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:˙˙˙˙˙˙˙˙˙dŹ
 
_user_specified_nameinputs
űż
Ż

__inference_call_3827796

inputs
context_sequence,
attention_head_2_3827588:
ŹŹ,
attention_head_2_3827590:
ŹŹ,
attention_head_2_3827592:
ŹŹA
2layer_normalization_1_cast_readvariableop_resource:	ŹC
4layer_normalization_1_cast_1_readvariableop_resource:	Ź,
attention_head_3_3827683:
ŹŹ,
attention_head_3_3827685:
ŹŹ,
attention_head_3_3827687:
ŹŹJ
6sequential_1_dense_2_tensordot_readvariableop_resource:
ŹŹC
4sequential_1_dense_2_biasadd_readvariableop_resource:	ŹJ
6sequential_1_dense_3_tensordot_readvariableop_resource:
ŹŹC
4sequential_1_dense_3_biasadd_readvariableop_resource:	Ź
identity˘(attention_head_2/StatefulPartitionedCall˘(attention_head_3/StatefulPartitionedCall˘)layer_normalization_1/Cast/ReadVariableOp˘+layer_normalization_1/Cast_1/ReadVariableOp˘+layer_normalization_1/Cast_2/ReadVariableOp˘+layer_normalization_1/Cast_3/ReadVariableOp˘+layer_normalization_1/Cast_4/ReadVariableOp˘+layer_normalization_1/Cast_5/ReadVariableOp˘+sequential_1/dense_2/BiasAdd/ReadVariableOp˘-sequential_1/dense_2/Tensordot/ReadVariableOp˘+sequential_1/dense_3/BiasAdd/ReadVariableOp˘-sequential_1/dense_3/Tensordot/ReadVariableOp
(attention_head_2/StatefulPartitionedCallStatefulPartitionedCallinputsinputsinputsattention_head_2_3827588attention_head_2_3827590attention_head_2_3827592*
Tin

2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:ddŹ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *!
fR
__inference_call_3827587u
addAddV21attention_head_2/StatefulPartitionedCall:output:0inputs*
T0*#
_output_shapes
:ddŹp
layer_normalization_1/ShapeConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ,  s
)layer_normalization_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+layer_normalization_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+layer_normalization_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ż
#layer_normalization_1/strided_sliceStridedSlice$layer_normalization_1/Shape:output:02layer_normalization_1/strided_slice/stack:output:04layer_normalization_1/strided_slice/stack_1:output:04layer_normalization_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
layer_normalization_1/mul/xConst*
_output_shapes
: *
dtype0*
value	B :
layer_normalization_1/mulMul$layer_normalization_1/mul/x:output:0,layer_normalization_1/strided_slice:output:0*
T0*
_output_shapes
: u
+layer_normalization_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ç
%layer_normalization_1/strided_slice_1StridedSlice$layer_normalization_1/Shape:output:04layer_normalization_1/strided_slice_1/stack:output:06layer_normalization_1/strided_slice_1/stack_1:output:06layer_normalization_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
layer_normalization_1/mul_1Mullayer_normalization_1/mul:z:0.layer_normalization_1/strided_slice_1:output:0*
T0*
_output_shapes
: u
+layer_normalization_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ç
%layer_normalization_1/strided_slice_2StridedSlice$layer_normalization_1/Shape:output:04layer_normalization_1/strided_slice_2/stack:output:06layer_normalization_1/strided_slice_2/stack_1:output:06layer_normalization_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
layer_normalization_1/mul_2/xConst*
_output_shapes
: *
dtype0*
value	B :
layer_normalization_1/mul_2Mul&layer_normalization_1/mul_2/x:output:0.layer_normalization_1/strided_slice_2:output:0*
T0*
_output_shapes
: g
%layer_normalization_1/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :g
%layer_normalization_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :ű
#layer_normalization_1/Reshape/shapePack.layer_normalization_1/Reshape/shape/0:output:0layer_normalization_1/mul_1:z:0layer_normalization_1/mul_2:z:0.layer_normalization_1/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:
layer_normalization_1/ReshapeReshapeadd:z:0,layer_normalization_1/Reshape/shape:output:0*
T0*(
_output_shapes
:NŹx
!layer_normalization_1/ones/packedPacklayer_normalization_1/mul_1:z:0*
N*
T0*
_output_shapes
:e
 layer_normalization_1/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
layer_normalization_1/onesFill*layer_normalization_1/ones/packed:output:0)layer_normalization_1/ones/Const:output:0*
T0*
_output_shapes	
:Ny
"layer_normalization_1/zeros/packedPacklayer_normalization_1/mul_1:z:0*
N*
T0*
_output_shapes
:f
!layer_normalization_1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ˘
layer_normalization_1/zerosFill+layer_normalization_1/zeros/packed:output:0*layer_normalization_1/zeros/Const:output:0*
T0*
_output_shapes	
:N^
layer_normalization_1/ConstConst*
_output_shapes
: *
dtype0*
valueB `
layer_normalization_1/Const_1Const*
_output_shapes
: *
dtype0*
valueB 
&layer_normalization_1/FusedBatchNormV3FusedBatchNormV3&layer_normalization_1/Reshape:output:0#layer_normalization_1/ones:output:0$layer_normalization_1/zeros:output:0$layer_normalization_1/Const:output:0&layer_normalization_1/Const_1:output:0*
T0*
U0*H
_output_shapes6
4:NŹ:N:N:N:N:*
data_formatNCHW*
epsilon%o:Ş
layer_normalization_1/Reshape_1Reshape*layer_normalization_1/FusedBatchNormV3:y:0$layer_normalization_1/Shape:output:0*
T0*#
_output_shapes
:ddŹ
)layer_normalization_1/Cast/ReadVariableOpReadVariableOp2layer_normalization_1_cast_readvariableop_resource*
_output_shapes	
:Ź*
dtype0­
layer_normalization_1/mul_3Mul(layer_normalization_1/Reshape_1:output:01layer_normalization_1/Cast/ReadVariableOp:value:0*
T0*#
_output_shapes
:ddŹ
+layer_normalization_1/Cast_1/ReadVariableOpReadVariableOp4layer_normalization_1_cast_1_readvariableop_resource*
_output_shapes	
:Ź*
dtype0Ś
layer_normalization_1/addAddV2layer_normalization_1/mul_3:z:03layer_normalization_1/Cast_1/ReadVariableOp:value:0*
T0*#
_output_shapes
:ddŹŔ
(attention_head_3/StatefulPartitionedCallStatefulPartitionedCalllayer_normalization_1/add:z:0layer_normalization_1/add:z:0context_sequenceattention_head_3_3827683attention_head_3_3827685attention_head_3_3827687*
Tin

2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:ddŹ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *!
fR
__inference_call_3827682
add_1AddV21attention_head_3/StatefulPartitionedCall:output:0layer_normalization_1/add:z:0*
T0*#
_output_shapes
:ddŹr
layer_normalization_1/Shape_1Const*
_output_shapes
:*
dtype0*!
valueB"d   d   ,  u
+layer_normalization_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-layer_normalization_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:É
%layer_normalization_1/strided_slice_3StridedSlice&layer_normalization_1/Shape_1:output:04layer_normalization_1/strided_slice_3/stack:output:06layer_normalization_1/strided_slice_3/stack_1:output:06layer_normalization_1/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
layer_normalization_1/mul_4/xConst*
_output_shapes
: *
dtype0*
value	B :
layer_normalization_1/mul_4Mul&layer_normalization_1/mul_4/x:output:0.layer_normalization_1/strided_slice_3:output:0*
T0*
_output_shapes
: u
+layer_normalization_1/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_1/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_1/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:É
%layer_normalization_1/strided_slice_4StridedSlice&layer_normalization_1/Shape_1:output:04layer_normalization_1/strided_slice_4/stack:output:06layer_normalization_1/strided_slice_4/stack_1:output:06layer_normalization_1/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
layer_normalization_1/mul_5Mullayer_normalization_1/mul_4:z:0.layer_normalization_1/strided_slice_4:output:0*
T0*
_output_shapes
: u
+layer_normalization_1/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_1/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_1/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:É
%layer_normalization_1/strided_slice_5StridedSlice&layer_normalization_1/Shape_1:output:04layer_normalization_1/strided_slice_5/stack:output:06layer_normalization_1/strided_slice_5/stack_1:output:06layer_normalization_1/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
layer_normalization_1/mul_6/xConst*
_output_shapes
: *
dtype0*
value	B :
layer_normalization_1/mul_6Mul&layer_normalization_1/mul_6/x:output:0.layer_normalization_1/strided_slice_5:output:0*
T0*
_output_shapes
: i
'layer_normalization_1/Reshape_2/shape/0Const*
_output_shapes
: *
dtype0*
value	B :i
'layer_normalization_1/Reshape_2/shape/3Const*
_output_shapes
: *
dtype0*
value	B :
%layer_normalization_1/Reshape_2/shapePack0layer_normalization_1/Reshape_2/shape/0:output:0layer_normalization_1/mul_5:z:0layer_normalization_1/mul_6:z:00layer_normalization_1/Reshape_2/shape/3:output:0*
N*
T0*
_output_shapes
:
layer_normalization_1/Reshape_2Reshape	add_1:z:0.layer_normalization_1/Reshape_2/shape:output:0*
T0*(
_output_shapes
:NŹz
#layer_normalization_1/ones_1/packedPacklayer_normalization_1/mul_5:z:0*
N*
T0*
_output_shapes
:g
"layer_normalization_1/ones_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ľ
layer_normalization_1/ones_1Fill,layer_normalization_1/ones_1/packed:output:0+layer_normalization_1/ones_1/Const:output:0*
T0*
_output_shapes	
:N{
$layer_normalization_1/zeros_1/packedPacklayer_normalization_1/mul_5:z:0*
N*
T0*
_output_shapes
:h
#layer_normalization_1/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ¨
layer_normalization_1/zeros_1Fill-layer_normalization_1/zeros_1/packed:output:0,layer_normalization_1/zeros_1/Const:output:0*
T0*
_output_shapes	
:N`
layer_normalization_1/Const_2Const*
_output_shapes
: *
dtype0*
valueB `
layer_normalization_1/Const_3Const*
_output_shapes
: *
dtype0*
valueB 
(layer_normalization_1/FusedBatchNormV3_1FusedBatchNormV3(layer_normalization_1/Reshape_2:output:0%layer_normalization_1/ones_1:output:0&layer_normalization_1/zeros_1:output:0&layer_normalization_1/Const_2:output:0&layer_normalization_1/Const_3:output:0*
T0*
U0*H
_output_shapes6
4:NŹ:N:N:N:N:*
data_formatNCHW*
epsilon%o:Ž
layer_normalization_1/Reshape_3Reshape,layer_normalization_1/FusedBatchNormV3_1:y:0&layer_normalization_1/Shape_1:output:0*
T0*#
_output_shapes
:ddŹ
+layer_normalization_1/Cast_2/ReadVariableOpReadVariableOp2layer_normalization_1_cast_readvariableop_resource*
_output_shapes	
:Ź*
dtype0Ż
layer_normalization_1/mul_7Mul(layer_normalization_1/Reshape_3:output:03layer_normalization_1/Cast_2/ReadVariableOp:value:0*
T0*#
_output_shapes
:ddŹ
+layer_normalization_1/Cast_3/ReadVariableOpReadVariableOp4layer_normalization_1_cast_1_readvariableop_resource*
_output_shapes	
:Ź*
dtype0¨
layer_normalization_1/add_1AddV2layer_normalization_1/mul_7:z:03layer_normalization_1/Cast_3/ReadVariableOp:value:0*
T0*#
_output_shapes
:ddŹŚ
-sequential_1/dense_2/Tensordot/ReadVariableOpReadVariableOp6sequential_1_dense_2_tensordot_readvariableop_resource* 
_output_shapes
:
ŹŹ*
dtype0}
,sequential_1/dense_2/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"'  ,  ´
&sequential_1/dense_2/Tensordot/ReshapeReshapelayer_normalization_1/add_1:z:05sequential_1/dense_2/Tensordot/Reshape/shape:output:0*
T0* 
_output_shapes
:
NŹÂ
%sequential_1/dense_2/Tensordot/MatMulMatMul/sequential_1/dense_2/Tensordot/Reshape:output:05sequential_1/dense_2/Tensordot/ReadVariableOp:value:0*
T0* 
_output_shapes
:
NŹy
$sequential_1/dense_2/Tensordot/shapeConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ,  ˇ
sequential_1/dense_2/TensordotReshape/sequential_1/dense_2/Tensordot/MatMul:product:0-sequential_1/dense_2/Tensordot/shape:output:0*
T0*#
_output_shapes
:ddŹ
+sequential_1/dense_2/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:Ź*
dtype0ł
sequential_1/dense_2/BiasAddBiasAdd'sequential_1/dense_2/Tensordot:output:03sequential_1/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*#
_output_shapes
:ddŹw
sequential_1/dense_2/LeakyRelu	LeakyRelu%sequential_1/dense_2/BiasAdd:output:0*#
_output_shapes
:ddŹŚ
-sequential_1/dense_3/Tensordot/ReadVariableOpReadVariableOp6sequential_1_dense_3_tensordot_readvariableop_resource* 
_output_shapes
:
ŹŹ*
dtype0}
,sequential_1/dense_3/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"'  ,  Á
&sequential_1/dense_3/Tensordot/ReshapeReshape,sequential_1/dense_2/LeakyRelu:activations:05sequential_1/dense_3/Tensordot/Reshape/shape:output:0*
T0* 
_output_shapes
:
NŹÂ
%sequential_1/dense_3/Tensordot/MatMulMatMul/sequential_1/dense_3/Tensordot/Reshape:output:05sequential_1/dense_3/Tensordot/ReadVariableOp:value:0*
T0* 
_output_shapes
:
NŹy
$sequential_1/dense_3/Tensordot/shapeConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ,  ˇ
sequential_1/dense_3/TensordotReshape/sequential_1/dense_3/Tensordot/MatMul:product:0-sequential_1/dense_3/Tensordot/shape:output:0*
T0*#
_output_shapes
:ddŹ
+sequential_1/dense_3/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_3_biasadd_readvariableop_resource*
_output_shapes	
:Ź*
dtype0ł
sequential_1/dense_3/BiasAddBiasAdd'sequential_1/dense_3/Tensordot:output:03sequential_1/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*#
_output_shapes
:ddŹ
add_2AddV2%sequential_1/dense_3/BiasAdd:output:0layer_normalization_1/add_1:z:0*
T0*#
_output_shapes
:ddŹr
layer_normalization_1/Shape_2Const*
_output_shapes
:*
dtype0*!
valueB"d   d   ,  u
+layer_normalization_1/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-layer_normalization_1/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_1/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB:É
%layer_normalization_1/strided_slice_6StridedSlice&layer_normalization_1/Shape_2:output:04layer_normalization_1/strided_slice_6/stack:output:06layer_normalization_1/strided_slice_6/stack_1:output:06layer_normalization_1/strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
layer_normalization_1/mul_8/xConst*
_output_shapes
: *
dtype0*
value	B :
layer_normalization_1/mul_8Mul&layer_normalization_1/mul_8/x:output:0.layer_normalization_1/strided_slice_6:output:0*
T0*
_output_shapes
: u
+layer_normalization_1/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_1/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_1/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB:É
%layer_normalization_1/strided_slice_7StridedSlice&layer_normalization_1/Shape_2:output:04layer_normalization_1/strided_slice_7/stack:output:06layer_normalization_1/strided_slice_7/stack_1:output:06layer_normalization_1/strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
layer_normalization_1/mul_9Mullayer_normalization_1/mul_8:z:0.layer_normalization_1/strided_slice_7:output:0*
T0*
_output_shapes
: u
+layer_normalization_1/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_1/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_1/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:É
%layer_normalization_1/strided_slice_8StridedSlice&layer_normalization_1/Shape_2:output:04layer_normalization_1/strided_slice_8/stack:output:06layer_normalization_1/strided_slice_8/stack_1:output:06layer_normalization_1/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask`
layer_normalization_1/mul_10/xConst*
_output_shapes
: *
dtype0*
value	B :
layer_normalization_1/mul_10Mul'layer_normalization_1/mul_10/x:output:0.layer_normalization_1/strided_slice_8:output:0*
T0*
_output_shapes
: i
'layer_normalization_1/Reshape_4/shape/0Const*
_output_shapes
: *
dtype0*
value	B :i
'layer_normalization_1/Reshape_4/shape/3Const*
_output_shapes
: *
dtype0*
value	B :
%layer_normalization_1/Reshape_4/shapePack0layer_normalization_1/Reshape_4/shape/0:output:0layer_normalization_1/mul_9:z:0 layer_normalization_1/mul_10:z:00layer_normalization_1/Reshape_4/shape/3:output:0*
N*
T0*
_output_shapes
:
layer_normalization_1/Reshape_4Reshape	add_2:z:0.layer_normalization_1/Reshape_4/shape:output:0*
T0*(
_output_shapes
:NŹz
#layer_normalization_1/ones_2/packedPacklayer_normalization_1/mul_9:z:0*
N*
T0*
_output_shapes
:g
"layer_normalization_1/ones_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ľ
layer_normalization_1/ones_2Fill,layer_normalization_1/ones_2/packed:output:0+layer_normalization_1/ones_2/Const:output:0*
T0*
_output_shapes	
:N{
$layer_normalization_1/zeros_2/packedPacklayer_normalization_1/mul_9:z:0*
N*
T0*
_output_shapes
:h
#layer_normalization_1/zeros_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ¨
layer_normalization_1/zeros_2Fill-layer_normalization_1/zeros_2/packed:output:0,layer_normalization_1/zeros_2/Const:output:0*
T0*
_output_shapes	
:N`
layer_normalization_1/Const_4Const*
_output_shapes
: *
dtype0*
valueB `
layer_normalization_1/Const_5Const*
_output_shapes
: *
dtype0*
valueB 
(layer_normalization_1/FusedBatchNormV3_2FusedBatchNormV3(layer_normalization_1/Reshape_4:output:0%layer_normalization_1/ones_2:output:0&layer_normalization_1/zeros_2:output:0&layer_normalization_1/Const_4:output:0&layer_normalization_1/Const_5:output:0*
T0*
U0*H
_output_shapes6
4:NŹ:N:N:N:N:*
data_formatNCHW*
epsilon%o:Ž
layer_normalization_1/Reshape_5Reshape,layer_normalization_1/FusedBatchNormV3_2:y:0&layer_normalization_1/Shape_2:output:0*
T0*#
_output_shapes
:ddŹ
+layer_normalization_1/Cast_4/ReadVariableOpReadVariableOp2layer_normalization_1_cast_readvariableop_resource*
_output_shapes	
:Ź*
dtype0°
layer_normalization_1/mul_11Mul(layer_normalization_1/Reshape_5:output:03layer_normalization_1/Cast_4/ReadVariableOp:value:0*
T0*#
_output_shapes
:ddŹ
+layer_normalization_1/Cast_5/ReadVariableOpReadVariableOp4layer_normalization_1_cast_1_readvariableop_resource*
_output_shapes	
:Ź*
dtype0Š
layer_normalization_1/add_2AddV2 layer_normalization_1/mul_11:z:03layer_normalization_1/Cast_5/ReadVariableOp:value:0*
T0*#
_output_shapes
:ddŹj
IdentityIdentitylayer_normalization_1/add_2:z:0^NoOp*
T0*#
_output_shapes
:ddŹę
NoOpNoOp)^attention_head_2/StatefulPartitionedCall)^attention_head_3/StatefulPartitionedCall*^layer_normalization_1/Cast/ReadVariableOp,^layer_normalization_1/Cast_1/ReadVariableOp,^layer_normalization_1/Cast_2/ReadVariableOp,^layer_normalization_1/Cast_3/ReadVariableOp,^layer_normalization_1/Cast_4/ReadVariableOp,^layer_normalization_1/Cast_5/ReadVariableOp,^sequential_1/dense_2/BiasAdd/ReadVariableOp.^sequential_1/dense_2/Tensordot/ReadVariableOp,^sequential_1/dense_3/BiasAdd/ReadVariableOp.^sequential_1/dense_3/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:ddŹ:	dŹ: : : : : : : : : : : : 2T
(attention_head_2/StatefulPartitionedCall(attention_head_2/StatefulPartitionedCall2T
(attention_head_3/StatefulPartitionedCall(attention_head_3/StatefulPartitionedCall2V
)layer_normalization_1/Cast/ReadVariableOp)layer_normalization_1/Cast/ReadVariableOp2Z
+layer_normalization_1/Cast_1/ReadVariableOp+layer_normalization_1/Cast_1/ReadVariableOp2Z
+layer_normalization_1/Cast_2/ReadVariableOp+layer_normalization_1/Cast_2/ReadVariableOp2Z
+layer_normalization_1/Cast_3/ReadVariableOp+layer_normalization_1/Cast_3/ReadVariableOp2Z
+layer_normalization_1/Cast_4/ReadVariableOp+layer_normalization_1/Cast_4/ReadVariableOp2Z
+layer_normalization_1/Cast_5/ReadVariableOp+layer_normalization_1/Cast_5/ReadVariableOp2Z
+sequential_1/dense_2/BiasAdd/ReadVariableOp+sequential_1/dense_2/BiasAdd/ReadVariableOp2^
-sequential_1/dense_2/Tensordot/ReadVariableOp-sequential_1/dense_2/Tensordot/ReadVariableOp2Z
+sequential_1/dense_3/BiasAdd/ReadVariableOp+sequential_1/dense_3/BiasAdd/ReadVariableOp2^
-sequential_1/dense_3/Tensordot/ReadVariableOp-sequential_1/dense_3/Tensordot/ReadVariableOp:K G
#
_output_shapes
:ddŹ
 
_user_specified_nameinputs:QM

_output_shapes
:	dŹ
*
_user_specified_namecontext_sequence
ä
Ł
.__inference_sequential_2_layer_call_fn_3832424
dense_4_input
unknown:	Ź
	unknown_0:
identity˘StatefulPartitionedCallĺ
StatefulPartitionedCallStatefulPartitionedCalldense_4_inputunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sequential_2_layer_call_and_return_conditional_losses_3832417o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:˙˙˙˙˙˙˙˙˙Ź: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ź
'
_user_specified_namedense_4_input
Ř
Ó
9__inference_emotion_detection_model_layer_call_fn_3833882
x_in
unknown:	dŹ
	unknown_0:
XŹ
	unknown_1:
ŹŹ
	unknown_2:
ŹŹ
	unknown_3:
ŹŹ
	unknown_4:	Ź
	unknown_5:	Ź
	unknown_6:
ŹŹ
	unknown_7:	Ź
	unknown_8:
ŹŹ
	unknown_9:	Ź

unknown_10:
ŹŹ

unknown_11:
ŹŹ

unknown_12:
ŹŹ

unknown_13:	Ź

unknown_14:	Ź

unknown_15:
ŹŹ

unknown_16:
ŹŹ

unknown_17:
ŹŹ

unknown_18:
ŹŹ

unknown_19:	Ź

unknown_20:
ŹŹ

unknown_21:	Ź

unknown_22:	Ź

unknown_23:
identity˘StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallx_inunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23*%
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*;
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_emotion_detection_model_layer_call_and_return_conditional_losses_3833483f
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:˙˙˙˙˙˙˙˙˙: : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:I E
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_namex_in
Í?

T__inference_emotion_detection_model_layer_call_and_return_conditional_losses_3833957
x_inT
Atoken_and_position_embedding_embedding_1_embedding_lookup_3833889:	dŹS
?token_and_position_embedding_embedding_embedding_lookup_3833894:
XŹ-
transformer_block_3833900:
ŹŹ-
transformer_block_3833902:
ŹŹ-
transformer_block_3833904:
ŹŹ(
transformer_block_3833906:	Ź(
transformer_block_3833908:	Ź-
transformer_block_3833910:
ŹŹ(
transformer_block_3833912:	Ź-
transformer_block_3833914:
ŹŹ(
transformer_block_3833916:	Ź/
transformer_block_1_3833920:
ŹŹ/
transformer_block_1_3833922:
ŹŹ/
transformer_block_1_3833924:
ŹŹ*
transformer_block_1_3833926:	Ź*
transformer_block_1_3833928:	Ź/
transformer_block_1_3833930:
ŹŹ/
transformer_block_1_3833932:
ŹŹ/
transformer_block_1_3833934:
ŹŹ/
transformer_block_1_3833936:
ŹŹ*
transformer_block_1_3833938:	Ź/
transformer_block_1_3833940:
ŹŹ*
transformer_block_1_3833942:	ŹF
3sequential_2_dense_4_matmul_readvariableop_resource:	ŹB
4sequential_2_dense_4_biasadd_readvariableop_resource:
identity˘+sequential_2/dense_4/BiasAdd/ReadVariableOp˘*sequential_2/dense_4/MatMul/ReadVariableOp˘7token_and_position_embedding/embedding/embedding_lookup˘9token_and_position_embedding/embedding_1/embedding_lookup˘)transformer_block/StatefulPartitionedCall˘+transformer_block_1/StatefulPartitionedCallj
(token_and_position_embedding/range/startConst*
_output_shapes
: *
dtype0*
value	B : j
(token_and_position_embedding/range/limitConst*
_output_shapes
: *
dtype0*
value	B :dj
(token_and_position_embedding/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :ŕ
"token_and_position_embedding/rangeRange1token_and_position_embedding/range/start:output:01token_and_position_embedding/range/limit:output:01token_and_position_embedding/range/delta:output:0*
_output_shapes
:dĎ
9token_and_position_embedding/embedding_1/embedding_lookupResourceGatherAtoken_and_position_embedding_embedding_1_embedding_lookup_3833889+token_and_position_embedding/range:output:0*
Tindices0*T
_classJ
HFloc:@token_and_position_embedding/embedding_1/embedding_lookup/3833889*
_output_shapes
:	dŹ*
dtype0
Btoken_and_position_embedding/embedding_1/embedding_lookup/IdentityIdentityBtoken_and_position_embedding/embedding_1/embedding_lookup:output:0*
T0*T
_classJ
HFloc:@token_and_position_embedding/embedding_1/embedding_lookup/3833889*
_output_shapes
:	dŹÇ
Dtoken_and_position_embedding/embedding_1/embedding_lookup/Identity_1IdentityKtoken_and_position_embedding/embedding_1/embedding_lookup/Identity:output:0*
T0*
_output_shapes
:	dŹŤ
7token_and_position_embedding/embedding/embedding_lookupResourceGather?token_and_position_embedding_embedding_embedding_lookup_3833894x_in*
Tindices0*R
_classH
FDloc:@token_and_position_embedding/embedding/embedding_lookup/3833894*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ź*
dtype0
@token_and_position_embedding/embedding/embedding_lookup/IdentityIdentity@token_and_position_embedding/embedding/embedding_lookup:output:0*
T0*R
_classH
FDloc:@token_and_position_embedding/embedding/embedding_lookup/3833894*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ŹĚ
Btoken_and_position_embedding/embedding/embedding_lookup/Identity_1IdentityItoken_and_position_embedding/embedding/embedding_lookup/Identity:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Źď
 token_and_position_embedding/addAddV2Ktoken_and_position_embedding/embedding/embedding_lookup/Identity_1:output:0Mtoken_and_position_embedding/embedding_1/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:	dŹĆ
)transformer_block/StatefulPartitionedCallStatefulPartitionedCall$token_and_position_embedding/add:z:0transformer_block_3833900transformer_block_3833902transformer_block_3833904transformer_block_3833906transformer_block_3833908transformer_block_3833910transformer_block_3833912transformer_block_3833914transformer_block_3833916*
Tin
2
*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:ddŹ*+
_read_only_resource_inputs
		*-
config_proto

CPU

GPU 2J 8 *!
fR
__inference_call_3827515~
dropout/IdentityIdentity2transformer_block/StatefulPartitionedCall:output:0*
T0*#
_output_shapes
:ddŹÓ
+transformer_block_1/StatefulPartitionedCallStatefulPartitionedCalldropout/Identity:output:0$token_and_position_embedding/add:z:0transformer_block_1_3833920transformer_block_1_3833922transformer_block_1_3833924transformer_block_1_3833926transformer_block_1_3833928transformer_block_1_3833930transformer_block_1_3833932transformer_block_1_3833934transformer_block_1_3833936transformer_block_1_3833938transformer_block_1_3833940transformer_block_1_3833942*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:ddŹ*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *!
fR
__inference_call_3827796P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 

ExpandDims
ExpandDims4transformer_block_1/StatefulPartitionedCall:output:0ExpandDims/dim:output:0*
T0*'
_output_shapes
:ddŹ{
*global_max_pooling2d/Max/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      
global_max_pooling2d/MaxMaxExpandDims:output:03global_max_pooling2d/Max/reduction_indices:output:0*
T0*
_output_shapes
:	Ź
*sequential_2/dense_4/MatMul/ReadVariableOpReadVariableOp3sequential_2_dense_4_matmul_readvariableop_resource*
_output_shapes
:	Ź*
dtype0Ľ
sequential_2/dense_4/MatMulMatMul!global_max_pooling2d/Max:output:02sequential_2/dense_4/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:
+sequential_2/dense_4/BiasAdd/ReadVariableOpReadVariableOp4sequential_2_dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ź
sequential_2/dense_4/BiasAddBiasAdd%sequential_2/dense_4/MatMul:product:03sequential_2/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:w
sequential_2/dense_4/SigmoidSigmoid%sequential_2/dense_4/BiasAdd:output:0*
T0*
_output_shapes

:f
IdentityIdentity sequential_2/dense_4/Sigmoid:y:0^NoOp*
T0*
_output_shapes

:ń
NoOpNoOp,^sequential_2/dense_4/BiasAdd/ReadVariableOp+^sequential_2/dense_4/MatMul/ReadVariableOp8^token_and_position_embedding/embedding/embedding_lookup:^token_and_position_embedding/embedding_1/embedding_lookup*^transformer_block/StatefulPartitionedCall,^transformer_block_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:˙˙˙˙˙˙˙˙˙: : : : : : : : : : : : : : : : : : : : : : : : : 2Z
+sequential_2/dense_4/BiasAdd/ReadVariableOp+sequential_2/dense_4/BiasAdd/ReadVariableOp2X
*sequential_2/dense_4/MatMul/ReadVariableOp*sequential_2/dense_4/MatMul/ReadVariableOp2r
7token_and_position_embedding/embedding/embedding_lookup7token_and_position_embedding/embedding/embedding_lookup2v
9token_and_position_embedding/embedding_1/embedding_lookup9token_and_position_embedding/embedding_1/embedding_lookup2V
)transformer_block/StatefulPartitionedCall)transformer_block/StatefulPartitionedCall2Z
+transformer_block_1/StatefulPartitionedCall+transformer_block_1/StatefulPartitionedCall:I E
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_namex_in
§

N__inference_transformer_block_layer_call_and_return_conditional_losses_3833325

inputs*
attention_head_3833210:
ŹŹ*
attention_head_3833212:
ŹŹ*
attention_head_3833214:
ŹŹ?
0layer_normalization_cast_readvariableop_resource:	ŹA
2layer_normalization_cast_1_readvariableop_resource:	ŹF
2sequential_dense_tensordot_readvariableop_resource:
ŹŹ?
0sequential_dense_biasadd_readvariableop_resource:	ŹH
4sequential_dense_1_tensordot_readvariableop_resource:
ŹŹA
2sequential_dense_1_biasadd_readvariableop_resource:	Ź
identity˘&attention_head/StatefulPartitionedCall˘'layer_normalization/Cast/ReadVariableOp˘)layer_normalization/Cast_1/ReadVariableOp˘)layer_normalization/Cast_2/ReadVariableOp˘)layer_normalization/Cast_3/ReadVariableOp˘'sequential/dense/BiasAdd/ReadVariableOp˘)sequential/dense/Tensordot/ReadVariableOp˘)sequential/dense_1/BiasAdd/ReadVariableOp˘+sequential/dense_1/Tensordot/ReadVariableOp
&attention_head/StatefulPartitionedCallStatefulPartitionedCallinputsinputsinputsattention_head_3833210attention_head_3833212attention_head_3833214*
Tin

2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:ddŹ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *!
fR
__inference_call_3827399s
addAddV2/attention_head/StatefulPartitionedCall:output:0inputs*
T0*#
_output_shapes
:ddŹn
layer_normalization/ShapeConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ,  q
'layer_normalization/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)layer_normalization/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)layer_normalization/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ľ
!layer_normalization/strided_sliceStridedSlice"layer_normalization/Shape:output:00layer_normalization/strided_slice/stack:output:02layer_normalization/strided_slice/stack_1:output:02layer_normalization/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask[
layer_normalization/mul/xConst*
_output_shapes
: *
dtype0*
value	B :
layer_normalization/mulMul"layer_normalization/mul/x:output:0*layer_normalization/strided_slice:output:0*
T0*
_output_shapes
: s
)layer_normalization/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:u
+layer_normalization/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+layer_normalization/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:˝
#layer_normalization/strided_slice_1StridedSlice"layer_normalization/Shape:output:02layer_normalization/strided_slice_1/stack:output:04layer_normalization/strided_slice_1/stack_1:output:04layer_normalization/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
layer_normalization/mul_1Mullayer_normalization/mul:z:0,layer_normalization/strided_slice_1:output:0*
T0*
_output_shapes
: s
)layer_normalization/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:u
+layer_normalization/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+layer_normalization/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:˝
#layer_normalization/strided_slice_2StridedSlice"layer_normalization/Shape:output:02layer_normalization/strided_slice_2/stack:output:04layer_normalization/strided_slice_2/stack_1:output:04layer_normalization/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
layer_normalization/mul_2/xConst*
_output_shapes
: *
dtype0*
value	B :
layer_normalization/mul_2Mul$layer_normalization/mul_2/x:output:0,layer_normalization/strided_slice_2:output:0*
T0*
_output_shapes
: e
#layer_normalization/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :e
#layer_normalization/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :ń
!layer_normalization/Reshape/shapePack,layer_normalization/Reshape/shape/0:output:0layer_normalization/mul_1:z:0layer_normalization/mul_2:z:0,layer_normalization/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:
layer_normalization/ReshapeReshapeadd:z:0*layer_normalization/Reshape/shape:output:0*
T0*(
_output_shapes
:NŹt
layer_normalization/ones/packedPacklayer_normalization/mul_1:z:0*
N*
T0*
_output_shapes
:c
layer_normalization/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
layer_normalization/onesFill(layer_normalization/ones/packed:output:0'layer_normalization/ones/Const:output:0*
T0*
_output_shapes	
:Nu
 layer_normalization/zeros/packedPacklayer_normalization/mul_1:z:0*
N*
T0*
_output_shapes
:d
layer_normalization/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
layer_normalization/zerosFill)layer_normalization/zeros/packed:output:0(layer_normalization/zeros/Const:output:0*
T0*
_output_shapes	
:N\
layer_normalization/ConstConst*
_output_shapes
: *
dtype0*
valueB ^
layer_normalization/Const_1Const*
_output_shapes
: *
dtype0*
valueB ô
$layer_normalization/FusedBatchNormV3FusedBatchNormV3$layer_normalization/Reshape:output:0!layer_normalization/ones:output:0"layer_normalization/zeros:output:0"layer_normalization/Const:output:0$layer_normalization/Const_1:output:0*
T0*
U0*H
_output_shapes6
4:NŹ:N:N:N:N:*
data_formatNCHW*
epsilon%o:¤
layer_normalization/Reshape_1Reshape(layer_normalization/FusedBatchNormV3:y:0"layer_normalization/Shape:output:0*
T0*#
_output_shapes
:ddŹ
'layer_normalization/Cast/ReadVariableOpReadVariableOp0layer_normalization_cast_readvariableop_resource*
_output_shapes	
:Ź*
dtype0§
layer_normalization/mul_3Mul&layer_normalization/Reshape_1:output:0/layer_normalization/Cast/ReadVariableOp:value:0*
T0*#
_output_shapes
:ddŹ
)layer_normalization/Cast_1/ReadVariableOpReadVariableOp2layer_normalization_cast_1_readvariableop_resource*
_output_shapes	
:Ź*
dtype0 
layer_normalization/addAddV2layer_normalization/mul_3:z:01layer_normalization/Cast_1/ReadVariableOp:value:0*
T0*#
_output_shapes
:ddŹ
)sequential/dense/Tensordot/ReadVariableOpReadVariableOp2sequential_dense_tensordot_readvariableop_resource* 
_output_shapes
:
ŹŹ*
dtype0y
(sequential/dense/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"'  ,  ¨
"sequential/dense/Tensordot/ReshapeReshapelayer_normalization/add:z:01sequential/dense/Tensordot/Reshape/shape:output:0*
T0* 
_output_shapes
:
NŹś
!sequential/dense/Tensordot/MatMulMatMul+sequential/dense/Tensordot/Reshape:output:01sequential/dense/Tensordot/ReadVariableOp:value:0*
T0* 
_output_shapes
:
NŹu
 sequential/dense/Tensordot/shapeConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ,  Ť
sequential/dense/TensordotReshape+sequential/dense/Tensordot/MatMul:product:0)sequential/dense/Tensordot/shape:output:0*
T0*#
_output_shapes
:ddŹ
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes	
:Ź*
dtype0§
sequential/dense/BiasAddBiasAdd#sequential/dense/Tensordot:output:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*#
_output_shapes
:ddŹo
sequential/dense/LeakyRelu	LeakyRelu!sequential/dense/BiasAdd:output:0*#
_output_shapes
:ddŹ˘
+sequential/dense_1/Tensordot/ReadVariableOpReadVariableOp4sequential_dense_1_tensordot_readvariableop_resource* 
_output_shapes
:
ŹŹ*
dtype0{
*sequential/dense_1/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"'  ,  š
$sequential/dense_1/Tensordot/ReshapeReshape(sequential/dense/LeakyRelu:activations:03sequential/dense_1/Tensordot/Reshape/shape:output:0*
T0* 
_output_shapes
:
NŹź
#sequential/dense_1/Tensordot/MatMulMatMul-sequential/dense_1/Tensordot/Reshape:output:03sequential/dense_1/Tensordot/ReadVariableOp:value:0*
T0* 
_output_shapes
:
NŹw
"sequential/dense_1/Tensordot/shapeConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ,  ą
sequential/dense_1/TensordotReshape-sequential/dense_1/Tensordot/MatMul:product:0+sequential/dense_1/Tensordot/shape:output:0*
T0*#
_output_shapes
:ddŹ
)sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:Ź*
dtype0­
sequential/dense_1/BiasAddBiasAdd%sequential/dense_1/Tensordot:output:01sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*#
_output_shapes
:ddŹ~
add_1AddV2#sequential/dense_1/BiasAdd:output:0layer_normalization/add:z:0*
T0*#
_output_shapes
:ddŹp
layer_normalization/Shape_1Const*
_output_shapes
:*
dtype0*!
valueB"d   d   ,  s
)layer_normalization/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+layer_normalization/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+layer_normalization/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ż
#layer_normalization/strided_slice_3StridedSlice$layer_normalization/Shape_1:output:02layer_normalization/strided_slice_3/stack:output:04layer_normalization/strided_slice_3/stack_1:output:04layer_normalization/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
layer_normalization/mul_4/xConst*
_output_shapes
: *
dtype0*
value	B :
layer_normalization/mul_4Mul$layer_normalization/mul_4/x:output:0,layer_normalization/strided_slice_3:output:0*
T0*
_output_shapes
: s
)layer_normalization/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:u
+layer_normalization/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+layer_normalization/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ż
#layer_normalization/strided_slice_4StridedSlice$layer_normalization/Shape_1:output:02layer_normalization/strided_slice_4/stack:output:04layer_normalization/strided_slice_4/stack_1:output:04layer_normalization/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
layer_normalization/mul_5Mullayer_normalization/mul_4:z:0,layer_normalization/strided_slice_4:output:0*
T0*
_output_shapes
: s
)layer_normalization/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:u
+layer_normalization/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+layer_normalization/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ż
#layer_normalization/strided_slice_5StridedSlice$layer_normalization/Shape_1:output:02layer_normalization/strided_slice_5/stack:output:04layer_normalization/strided_slice_5/stack_1:output:04layer_normalization/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
layer_normalization/mul_6/xConst*
_output_shapes
: *
dtype0*
value	B :
layer_normalization/mul_6Mul$layer_normalization/mul_6/x:output:0,layer_normalization/strided_slice_5:output:0*
T0*
_output_shapes
: g
%layer_normalization/Reshape_2/shape/0Const*
_output_shapes
: *
dtype0*
value	B :g
%layer_normalization/Reshape_2/shape/3Const*
_output_shapes
: *
dtype0*
value	B :÷
#layer_normalization/Reshape_2/shapePack.layer_normalization/Reshape_2/shape/0:output:0layer_normalization/mul_5:z:0layer_normalization/mul_6:z:0.layer_normalization/Reshape_2/shape/3:output:0*
N*
T0*
_output_shapes
:
layer_normalization/Reshape_2Reshape	add_1:z:0,layer_normalization/Reshape_2/shape:output:0*
T0*(
_output_shapes
:NŹv
!layer_normalization/ones_1/packedPacklayer_normalization/mul_5:z:0*
N*
T0*
_output_shapes
:e
 layer_normalization/ones_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
layer_normalization/ones_1Fill*layer_normalization/ones_1/packed:output:0)layer_normalization/ones_1/Const:output:0*
T0*
_output_shapes	
:Nw
"layer_normalization/zeros_1/packedPacklayer_normalization/mul_5:z:0*
N*
T0*
_output_shapes
:f
!layer_normalization/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ˘
layer_normalization/zeros_1Fill+layer_normalization/zeros_1/packed:output:0*layer_normalization/zeros_1/Const:output:0*
T0*
_output_shapes	
:N^
layer_normalization/Const_2Const*
_output_shapes
: *
dtype0*
valueB ^
layer_normalization/Const_3Const*
_output_shapes
: *
dtype0*
valueB ţ
&layer_normalization/FusedBatchNormV3_1FusedBatchNormV3&layer_normalization/Reshape_2:output:0#layer_normalization/ones_1:output:0$layer_normalization/zeros_1:output:0$layer_normalization/Const_2:output:0$layer_normalization/Const_3:output:0*
T0*
U0*H
_output_shapes6
4:NŹ:N:N:N:N:*
data_formatNCHW*
epsilon%o:¨
layer_normalization/Reshape_3Reshape*layer_normalization/FusedBatchNormV3_1:y:0$layer_normalization/Shape_1:output:0*
T0*#
_output_shapes
:ddŹ
)layer_normalization/Cast_2/ReadVariableOpReadVariableOp0layer_normalization_cast_readvariableop_resource*
_output_shapes	
:Ź*
dtype0Š
layer_normalization/mul_7Mul&layer_normalization/Reshape_3:output:01layer_normalization/Cast_2/ReadVariableOp:value:0*
T0*#
_output_shapes
:ddŹ
)layer_normalization/Cast_3/ReadVariableOpReadVariableOp2layer_normalization_cast_1_readvariableop_resource*
_output_shapes	
:Ź*
dtype0˘
layer_normalization/add_1AddV2layer_normalization/mul_7:z:01layer_normalization/Cast_3/ReadVariableOp:value:0*
T0*#
_output_shapes
:ddŹh
IdentityIdentitylayer_normalization/add_1:z:0^NoOp*
T0*#
_output_shapes
:ddŹÍ
NoOpNoOp'^attention_head/StatefulPartitionedCall(^layer_normalization/Cast/ReadVariableOp*^layer_normalization/Cast_1/ReadVariableOp*^layer_normalization/Cast_2/ReadVariableOp*^layer_normalization/Cast_3/ReadVariableOp(^sequential/dense/BiasAdd/ReadVariableOp*^sequential/dense/Tensordot/ReadVariableOp*^sequential/dense_1/BiasAdd/ReadVariableOp,^sequential/dense_1/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:	dŹ: : : : : : : : : 2P
&attention_head/StatefulPartitionedCall&attention_head/StatefulPartitionedCall2R
'layer_normalization/Cast/ReadVariableOp'layer_normalization/Cast/ReadVariableOp2V
)layer_normalization/Cast_1/ReadVariableOp)layer_normalization/Cast_1/ReadVariableOp2V
)layer_normalization/Cast_2/ReadVariableOp)layer_normalization/Cast_2/ReadVariableOp2V
)layer_normalization/Cast_3/ReadVariableOp)layer_normalization/Cast_3/ReadVariableOp2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2V
)sequential/dense/Tensordot/ReadVariableOp)sequential/dense/Tensordot/ReadVariableOp2V
)sequential/dense_1/BiasAdd/ReadVariableOp)sequential/dense_1/BiasAdd/ReadVariableOp2Z
+sequential/dense_1/Tensordot/ReadVariableOp+sequential/dense_1/Tensordot/ReadVariableOp:G C

_output_shapes
:	dŹ
 
_user_specified_nameinputs
Ů
ţ
D__inference_dense_3_layer_call_and_return_conditional_losses_3835297

inputs5
!tensordot_readvariableop_resource:
ŹŹ.
biasadd_readvariableop_resource:	Ź
identity˘BiasAdd/ReadVariableOp˘Tensordot/ReadVariableOp|
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource* 
_output_shapes
:
ŹŹ*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ť
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ż
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:z
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙dŹ
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ź\
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:ŹY
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : §
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙dŹs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:Ź*
dtype0}
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙dŹd
IdentityIdentityBiasAdd:output:0^NoOp*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙dŹz
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:˙˙˙˙˙˙˙˙˙dŹ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:˙˙˙˙˙˙˙˙˙dŹ
 
_user_specified_nameinputs
×

I__inference_sequential_2_layer_call_and_return_conditional_losses_3834812

inputs9
&dense_4_matmul_readvariableop_resource:	Ź5
'dense_4_biasadd_readvariableop_resource:
identity˘dense_4/BiasAdd/ReadVariableOp˘dense_4/MatMul/ReadVariableOp
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes
:	Ź*
dtype0y
dense_4/MatMulMatMulinputs%dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙f
dense_4/SigmoidSigmoiddense_4/BiasAdd:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙b
IdentityIdentitydense_4/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
NoOpNoOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:˙˙˙˙˙˙˙˙˙Ź: : 2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ź
 
_user_specified_nameinputs
Ř
Ó
9__inference_emotion_detection_model_layer_call_fn_3833827
x_in
unknown:	dŹ
	unknown_0:
XŹ
	unknown_1:
ŹŹ
	unknown_2:
ŹŹ
	unknown_3:
ŹŹ
	unknown_4:	Ź
	unknown_5:	Ź
	unknown_6:
ŹŹ
	unknown_7:	Ź
	unknown_8:
ŹŹ
	unknown_9:	Ź

unknown_10:
ŹŹ

unknown_11:
ŹŹ

unknown_12:
ŹŹ

unknown_13:	Ź

unknown_14:	Ź

unknown_15:
ŹŹ

unknown_16:
ŹŹ

unknown_17:
ŹŹ

unknown_18:
ŹŹ

unknown_19:	Ź

unknown_20:
ŹŹ

unknown_21:	Ź

unknown_22:	Ź

unknown_23:
identity˘StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallx_inunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23*%
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*;
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_emotion_detection_model_layer_call_and_return_conditional_losses_3832876f
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:˙˙˙˙˙˙˙˙˙: : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:I E
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_namex_in
ô	
í
3__inference_transformer_block_layer_call_fn_3834129

inputs
unknown:
ŹŹ
	unknown_0:
ŹŹ
	unknown_1:
ŹŹ
	unknown_2:	Ź
	unknown_3:	Ź
	unknown_4:
ŹŹ
	unknown_5:	Ź
	unknown_6:
ŹŹ
	unknown_7:	Ź
identity˘StatefulPartitionedCallş
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
Tin
2
*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:ddŹ*+
_read_only_resource_inputs
		*-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_transformer_block_layer_call_and_return_conditional_losses_3833325k
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*#
_output_shapes
:ddŹ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:	dŹ: : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:G C

_output_shapes
:	dŹ
 
_user_specified_nameinputs

Â
%__inference_signature_wrapper_3833772
input_1
unknown:	dŹ
	unknown_0:
XŹ
	unknown_1:
ŹŹ
	unknown_2:
ŹŹ
	unknown_3:
ŹŹ
	unknown_4:	Ź
	unknown_5:	Ź
	unknown_6:
ŹŹ
	unknown_7:	Ź
	unknown_8:
ŹŹ
	unknown_9:	Ź

unknown_10:
ŹŹ

unknown_11:
ŹŹ

unknown_12:
ŹŹ

unknown_13:	Ź

unknown_14:	Ź

unknown_15:
ŹŹ

unknown_16:
ŹŹ

unknown_17:
ŹŹ

unknown_18:
ŹŹ

unknown_19:	Ź

unknown_20:
ŹŹ

unknown_21:	Ź

unknown_22:	Ź

unknown_23:
identity˘StatefulPartitionedCallč
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23*%
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*;
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *+
f&R$
"__inference__wrapped_model_3831993f
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:˙˙˙˙˙˙˙˙˙: : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:L H
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
!
_user_specified_name	input_1
Ĺ

)__inference_dense_4_layer_call_fn_3835128

inputs
unknown:	Ź
	unknown_0:
identity˘StatefulPartitionedCallŮ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_4_layer_call_and_return_conditional_losses_3832410o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:˙˙˙˙˙˙˙˙˙Ź: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ź
 
_user_specified_nameinputs
ř>
Ţ
I__inference_sequential_1_layer_call_and_return_conditional_losses_3835119

inputs=
)dense_2_tensordot_readvariableop_resource:
ŹŹ6
'dense_2_biasadd_readvariableop_resource:	Ź=
)dense_3_tensordot_readvariableop_resource:
ŹŹ6
'dense_3_biasadd_readvariableop_resource:	Ź
identity˘dense_2/BiasAdd/ReadVariableOp˘ dense_2/Tensordot/ReadVariableOp˘dense_3/BiasAdd/ReadVariableOp˘ dense_3/Tensordot/ReadVariableOp
 dense_2/Tensordot/ReadVariableOpReadVariableOp)dense_2_tensordot_readvariableop_resource* 
_output_shapes
:
ŹŹ*
dtype0`
dense_2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:g
dense_2/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       M
dense_2/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:a
dense_2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : Ű
dense_2/Tensordot/GatherV2GatherV2 dense_2/Tensordot/Shape:output:0dense_2/Tensordot/free:output:0(dense_2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:c
!dense_2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ß
dense_2/Tensordot/GatherV2_1GatherV2 dense_2/Tensordot/Shape:output:0dense_2/Tensordot/axes:output:0*dense_2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense_2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
dense_2/Tensordot/ProdProd#dense_2/Tensordot/GatherV2:output:0 dense_2/Tensordot/Const:output:0*
T0*
_output_shapes
: c
dense_2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
dense_2/Tensordot/Prod_1Prod%dense_2/Tensordot/GatherV2_1:output:0"dense_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
dense_2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ź
dense_2/Tensordot/concatConcatV2dense_2/Tensordot/free:output:0dense_2/Tensordot/axes:output:0&dense_2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:
dense_2/Tensordot/stackPackdense_2/Tensordot/Prod:output:0!dense_2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:
dense_2/Tensordot/transpose	Transposeinputs!dense_2/Tensordot/concat:output:0*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙dŹ˘
dense_2/Tensordot/ReshapeReshapedense_2/Tensordot/transpose:y:0 dense_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙Ł
dense_2/Tensordot/MatMulMatMul"dense_2/Tensordot/Reshape:output:0(dense_2/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Źd
dense_2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Źa
dense_2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ç
dense_2/Tensordot/concat_1ConcatV2#dense_2/Tensordot/GatherV2:output:0"dense_2/Tensordot/Const_2:output:0(dense_2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
dense_2/TensordotReshape"dense_2/Tensordot/MatMul:product:0#dense_2/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙dŹ
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes	
:Ź*
dtype0
dense_2/BiasAddBiasAdddense_2/Tensordot:output:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙dŹf
dense_2/LeakyRelu	LeakyReludense_2/BiasAdd:output:0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙dŹ
 dense_3/Tensordot/ReadVariableOpReadVariableOp)dense_3_tensordot_readvariableop_resource* 
_output_shapes
:
ŹŹ*
dtype0`
dense_3/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:g
dense_3/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       f
dense_3/Tensordot/ShapeShapedense_2/LeakyRelu:activations:0*
T0*
_output_shapes
:a
dense_3/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : Ű
dense_3/Tensordot/GatherV2GatherV2 dense_3/Tensordot/Shape:output:0dense_3/Tensordot/free:output:0(dense_3/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:c
!dense_3/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ß
dense_3/Tensordot/GatherV2_1GatherV2 dense_3/Tensordot/Shape:output:0dense_3/Tensordot/axes:output:0*dense_3/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense_3/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
dense_3/Tensordot/ProdProd#dense_3/Tensordot/GatherV2:output:0 dense_3/Tensordot/Const:output:0*
T0*
_output_shapes
: c
dense_3/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
dense_3/Tensordot/Prod_1Prod%dense_3/Tensordot/GatherV2_1:output:0"dense_3/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
dense_3/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ź
dense_3/Tensordot/concatConcatV2dense_3/Tensordot/free:output:0dense_3/Tensordot/axes:output:0&dense_3/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:
dense_3/Tensordot/stackPackdense_3/Tensordot/Prod:output:0!dense_3/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:Ł
dense_3/Tensordot/transpose	Transposedense_2/LeakyRelu:activations:0!dense_3/Tensordot/concat:output:0*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙dŹ˘
dense_3/Tensordot/ReshapeReshapedense_3/Tensordot/transpose:y:0 dense_3/Tensordot/stack:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙Ł
dense_3/Tensordot/MatMulMatMul"dense_3/Tensordot/Reshape:output:0(dense_3/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Źd
dense_3/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Źa
dense_3/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ç
dense_3/Tensordot/concat_1ConcatV2#dense_3/Tensordot/GatherV2:output:0"dense_3/Tensordot/Const_2:output:0(dense_3/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
dense_3/TensordotReshape"dense_3/Tensordot/MatMul:product:0#dense_3/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙dŹ
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes	
:Ź*
dtype0
dense_3/BiasAddBiasAdddense_3/Tensordot:output:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙dŹl
IdentityIdentitydense_3/BiasAdd:output:0^NoOp*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙dŹÎ
NoOpNoOp^dense_2/BiasAdd/ReadVariableOp!^dense_2/Tensordot/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp!^dense_3/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :˙˙˙˙˙˙˙˙˙dŹ: : : : 2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2D
 dense_2/Tensordot/ReadVariableOp dense_2/Tensordot/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2D
 dense_3/Tensordot/ReadVariableOp dense_3/Tensordot/ReadVariableOp:T P
,
_output_shapes
:˙˙˙˙˙˙˙˙˙dŹ
 
_user_specified_nameinputs
Ĺ
˝
I__inference_sequential_1_layer_call_and_return_conditional_losses_3832379
dense_2_input#
dense_2_3832368:
ŹŹ
dense_2_3832370:	Ź#
dense_3_3832373:
ŹŹ
dense_3_3832375:	Ź
identity˘dense_2/StatefulPartitionedCall˘dense_3/StatefulPartitionedCallű
dense_2/StatefulPartitionedCallStatefulPartitionedCalldense_2_inputdense_2_3832368dense_2_3832370*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:˙˙˙˙˙˙˙˙˙dŹ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_2_layer_call_and_return_conditional_losses_3832224
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_3832373dense_3_3832375*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:˙˙˙˙˙˙˙˙˙dŹ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_3_layer_call_and_return_conditional_losses_3832260|
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙dŹ
NoOpNoOp ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :˙˙˙˙˙˙˙˙˙dŹ: : : : 2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall:[ W
,
_output_shapes
:˙˙˙˙˙˙˙˙˙dŹ
'
_user_specified_namedense_2_input
§
ł
G__inference_sequential_layer_call_and_return_conditional_losses_3832172
dense_input!
dense_3832161:
ŹŹ
dense_3832163:	Ź#
dense_1_3832166:
ŹŹ
dense_1_3832168:	Ź
identity˘dense/StatefulPartitionedCall˘dense_1/StatefulPartitionedCallń
dense/StatefulPartitionedCallStatefulPartitionedCalldense_inputdense_3832161dense_3832163*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:˙˙˙˙˙˙˙˙˙dŹ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_3832031
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_3832166dense_1_3832168*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:˙˙˙˙˙˙˙˙˙dŹ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_3832067|
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙dŹ
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :˙˙˙˙˙˙˙˙˙dŹ: : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:Y U
,
_output_shapes
:˙˙˙˙˙˙˙˙˙dŹ
%
_user_specified_namedense_input
Ő

'__inference_dense_layer_call_fn_3835148

inputs
unknown:
ŹŹ
	unknown_0:	Ź
identity˘StatefulPartitionedCallÜ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:˙˙˙˙˙˙˙˙˙dŹ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_3832031t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙dŹ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:˙˙˙˙˙˙˙˙˙dŹ: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:˙˙˙˙˙˙˙˙˙dŹ
 
_user_specified_nameinputs


c
D__inference_dropout_layer_call_and_return_conditional_losses_3834839

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @`
dropout/MulMulinputsdropout/Const:output:0*
T0*#
_output_shapes
:ddŹb
dropout/ShapeConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ,  
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*#
_output_shapes
:ddŹ*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?˘
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*#
_output_shapes
:ddŹT
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*#
_output_shapes
:ddŹ]
IdentityIdentitydropout/SelectV2:output:0*
T0*#
_output_shapes
:ddŹ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*"
_input_shapes
:ddŹ:K G
#
_output_shapes
:ddŹ
 
_user_specified_nameinputs
Ů

)__inference_dense_2_layer_call_fn_3835227

inputs
unknown:
ŹŹ
	unknown_0:	Ź
identity˘StatefulPartitionedCallŢ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:˙˙˙˙˙˙˙˙˙dŹ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_2_layer_call_and_return_conditional_losses_3832224t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙dŹ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:˙˙˙˙˙˙˙˙˙dŹ: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:˙˙˙˙˙˙˙˙˙dŹ
 
_user_specified_nameinputs
ď=
Ô
G__inference_sequential_layer_call_and_return_conditional_losses_3834979

inputs;
'dense_tensordot_readvariableop_resource:
ŹŹ4
%dense_biasadd_readvariableop_resource:	Ź=
)dense_1_tensordot_readvariableop_resource:
ŹŹ6
'dense_1_biasadd_readvariableop_resource:	Ź
identity˘dense/BiasAdd/ReadVariableOp˘dense/Tensordot/ReadVariableOp˘dense_1/BiasAdd/ReadVariableOp˘ dense_1/Tensordot/ReadVariableOp
dense/Tensordot/ReadVariableOpReadVariableOp'dense_tensordot_readvariableop_resource* 
_output_shapes
:
ŹŹ*
dtype0^
dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:e
dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       K
dense/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:_
dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : Ó
dense/Tensordot/GatherV2GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/free:output:0&dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ×
dense/Tensordot/GatherV2_1GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/axes:output:0(dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:_
dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
dense/Tensordot/ProdProd!dense/Tensordot/GatherV2:output:0dense/Tensordot/Const:output:0*
T0*
_output_shapes
: a
dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
dense/Tensordot/Prod_1Prod#dense/Tensordot/GatherV2_1:output:0 dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: ]
dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ´
dense/Tensordot/concatConcatV2dense/Tensordot/free:output:0dense/Tensordot/axes:output:0$dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:
dense/Tensordot/stackPackdense/Tensordot/Prod:output:0dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:
dense/Tensordot/transpose	Transposeinputsdense/Tensordot/concat:output:0*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙dŹ
dense/Tensordot/ReshapeReshapedense/Tensordot/transpose:y:0dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
dense/Tensordot/MatMulMatMul dense/Tensordot/Reshape:output:0&dense/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Źb
dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Ź_
dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ż
dense/Tensordot/concat_1ConcatV2!dense/Tensordot/GatherV2:output:0 dense/Tensordot/Const_2:output:0&dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
dense/TensordotReshape dense/Tensordot/MatMul:product:0!dense/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙dŹ
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:Ź*
dtype0
dense/BiasAddBiasAdddense/Tensordot:output:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙dŹb
dense/LeakyRelu	LeakyReludense/BiasAdd:output:0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙dŹ
 dense_1/Tensordot/ReadVariableOpReadVariableOp)dense_1_tensordot_readvariableop_resource* 
_output_shapes
:
ŹŹ*
dtype0`
dense_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:g
dense_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       d
dense_1/Tensordot/ShapeShapedense/LeakyRelu:activations:0*
T0*
_output_shapes
:a
dense_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : Ű
dense_1/Tensordot/GatherV2GatherV2 dense_1/Tensordot/Shape:output:0dense_1/Tensordot/free:output:0(dense_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:c
!dense_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ß
dense_1/Tensordot/GatherV2_1GatherV2 dense_1/Tensordot/Shape:output:0dense_1/Tensordot/axes:output:0*dense_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
dense_1/Tensordot/ProdProd#dense_1/Tensordot/GatherV2:output:0 dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: c
dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
dense_1/Tensordot/Prod_1Prod%dense_1/Tensordot/GatherV2_1:output:0"dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ź
dense_1/Tensordot/concatConcatV2dense_1/Tensordot/free:output:0dense_1/Tensordot/axes:output:0&dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:
dense_1/Tensordot/stackPackdense_1/Tensordot/Prod:output:0!dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:Ą
dense_1/Tensordot/transpose	Transposedense/LeakyRelu:activations:0!dense_1/Tensordot/concat:output:0*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙dŹ˘
dense_1/Tensordot/ReshapeReshapedense_1/Tensordot/transpose:y:0 dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙Ł
dense_1/Tensordot/MatMulMatMul"dense_1/Tensordot/Reshape:output:0(dense_1/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Źd
dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Źa
dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ç
dense_1/Tensordot/concat_1ConcatV2#dense_1/Tensordot/GatherV2:output:0"dense_1/Tensordot/Const_2:output:0(dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
dense_1/TensordotReshape"dense_1/Tensordot/MatMul:product:0#dense_1/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙dŹ
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:Ź*
dtype0
dense_1/BiasAddBiasAdddense_1/Tensordot:output:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙dŹl
IdentityIdentitydense_1/BiasAdd:output:0^NoOp*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙dŹĘ
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/Tensordot/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp!^dense_1/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :˙˙˙˙˙˙˙˙˙dŹ: : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2@
dense/Tensordot/ReadVariableOpdense/Tensordot/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2D
 dense_1/Tensordot/ReadVariableOp dense_1/Tensordot/ReadVariableOp:T P
,
_output_shapes
:˙˙˙˙˙˙˙˙˙dŹ
 
_user_specified_nameinputs
Ů

)__inference_dense_3_layer_call_fn_3835267

inputs
unknown:
ŹŹ
	unknown_0:	Ź
identity˘StatefulPartitionedCallŢ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:˙˙˙˙˙˙˙˙˙dŹ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_3_layer_call_and_return_conditional_losses_3832260t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙dŹ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:˙˙˙˙˙˙˙˙˙dŹ: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:˙˙˙˙˙˙˙˙˙dŹ
 
_user_specified_nameinputs
Ç
b
D__inference_dropout_layer_call_and_return_conditional_losses_3832671

inputs

identity_1J
IdentityIdentityinputs*
T0*#
_output_shapes
:ddŹW

Identity_1IdentityIdentity:output:0*
T0*#
_output_shapes
:ddŹ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*"
_input_shapes
:ddŹ:K G
#
_output_shapes
:ddŹ
 
_user_specified_nameinputs


ö
D__inference_dense_4_layer_call_and_return_conditional_losses_3832410

inputs1
matmul_readvariableop_resource:	Ź-
biasadd_readvariableop_resource:
identity˘BiasAdd/ReadVariableOp˘MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	Ź*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:˙˙˙˙˙˙˙˙˙Ź: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ź
 
_user_specified_nameinputs
ř>
Ţ
I__inference_sequential_1_layer_call_and_return_conditional_losses_3835062

inputs=
)dense_2_tensordot_readvariableop_resource:
ŹŹ6
'dense_2_biasadd_readvariableop_resource:	Ź=
)dense_3_tensordot_readvariableop_resource:
ŹŹ6
'dense_3_biasadd_readvariableop_resource:	Ź
identity˘dense_2/BiasAdd/ReadVariableOp˘ dense_2/Tensordot/ReadVariableOp˘dense_3/BiasAdd/ReadVariableOp˘ dense_3/Tensordot/ReadVariableOp
 dense_2/Tensordot/ReadVariableOpReadVariableOp)dense_2_tensordot_readvariableop_resource* 
_output_shapes
:
ŹŹ*
dtype0`
dense_2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:g
dense_2/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       M
dense_2/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:a
dense_2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : Ű
dense_2/Tensordot/GatherV2GatherV2 dense_2/Tensordot/Shape:output:0dense_2/Tensordot/free:output:0(dense_2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:c
!dense_2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ß
dense_2/Tensordot/GatherV2_1GatherV2 dense_2/Tensordot/Shape:output:0dense_2/Tensordot/axes:output:0*dense_2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense_2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
dense_2/Tensordot/ProdProd#dense_2/Tensordot/GatherV2:output:0 dense_2/Tensordot/Const:output:0*
T0*
_output_shapes
: c
dense_2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
dense_2/Tensordot/Prod_1Prod%dense_2/Tensordot/GatherV2_1:output:0"dense_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
dense_2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ź
dense_2/Tensordot/concatConcatV2dense_2/Tensordot/free:output:0dense_2/Tensordot/axes:output:0&dense_2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:
dense_2/Tensordot/stackPackdense_2/Tensordot/Prod:output:0!dense_2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:
dense_2/Tensordot/transpose	Transposeinputs!dense_2/Tensordot/concat:output:0*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙dŹ˘
dense_2/Tensordot/ReshapeReshapedense_2/Tensordot/transpose:y:0 dense_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙Ł
dense_2/Tensordot/MatMulMatMul"dense_2/Tensordot/Reshape:output:0(dense_2/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Źd
dense_2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Źa
dense_2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ç
dense_2/Tensordot/concat_1ConcatV2#dense_2/Tensordot/GatherV2:output:0"dense_2/Tensordot/Const_2:output:0(dense_2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
dense_2/TensordotReshape"dense_2/Tensordot/MatMul:product:0#dense_2/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙dŹ
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes	
:Ź*
dtype0
dense_2/BiasAddBiasAdddense_2/Tensordot:output:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙dŹf
dense_2/LeakyRelu	LeakyReludense_2/BiasAdd:output:0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙dŹ
 dense_3/Tensordot/ReadVariableOpReadVariableOp)dense_3_tensordot_readvariableop_resource* 
_output_shapes
:
ŹŹ*
dtype0`
dense_3/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:g
dense_3/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       f
dense_3/Tensordot/ShapeShapedense_2/LeakyRelu:activations:0*
T0*
_output_shapes
:a
dense_3/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : Ű
dense_3/Tensordot/GatherV2GatherV2 dense_3/Tensordot/Shape:output:0dense_3/Tensordot/free:output:0(dense_3/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:c
!dense_3/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ß
dense_3/Tensordot/GatherV2_1GatherV2 dense_3/Tensordot/Shape:output:0dense_3/Tensordot/axes:output:0*dense_3/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense_3/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
dense_3/Tensordot/ProdProd#dense_3/Tensordot/GatherV2:output:0 dense_3/Tensordot/Const:output:0*
T0*
_output_shapes
: c
dense_3/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
dense_3/Tensordot/Prod_1Prod%dense_3/Tensordot/GatherV2_1:output:0"dense_3/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
dense_3/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ź
dense_3/Tensordot/concatConcatV2dense_3/Tensordot/free:output:0dense_3/Tensordot/axes:output:0&dense_3/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:
dense_3/Tensordot/stackPackdense_3/Tensordot/Prod:output:0!dense_3/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:Ł
dense_3/Tensordot/transpose	Transposedense_2/LeakyRelu:activations:0!dense_3/Tensordot/concat:output:0*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙dŹ˘
dense_3/Tensordot/ReshapeReshapedense_3/Tensordot/transpose:y:0 dense_3/Tensordot/stack:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙Ł
dense_3/Tensordot/MatMulMatMul"dense_3/Tensordot/Reshape:output:0(dense_3/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Źd
dense_3/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Źa
dense_3/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ç
dense_3/Tensordot/concat_1ConcatV2#dense_3/Tensordot/GatherV2:output:0"dense_3/Tensordot/Const_2:output:0(dense_3/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
dense_3/TensordotReshape"dense_3/Tensordot/MatMul:product:0#dense_3/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙dŹ
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes	
:Ź*
dtype0
dense_3/BiasAddBiasAdddense_3/Tensordot:output:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙dŹl
IdentityIdentitydense_3/BiasAdd:output:0^NoOp*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙dŹÎ
NoOpNoOp^dense_2/BiasAdd/ReadVariableOp!^dense_2/Tensordot/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp!^dense_3/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :˙˙˙˙˙˙˙˙˙dŹ: : : : 2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2D
 dense_2/Tensordot/ReadVariableOp dense_2/Tensordot/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2D
 dense_3/Tensordot/ReadVariableOp dense_3/Tensordot/ReadVariableOp:T P
,
_output_shapes
:˙˙˙˙˙˙˙˙˙dŹ
 
_user_specified_nameinputs
ş
×
.__inference_sequential_1_layer_call_fn_3835005

inputs
unknown:
ŹŹ
	unknown_0:	Ź
	unknown_1:
ŹŹ
	unknown_2:	Ź
identity˘StatefulPartitionedCallý
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:˙˙˙˙˙˙˙˙˙dŹ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sequential_1_layer_call_and_return_conditional_losses_3832327t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙dŹ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :˙˙˙˙˙˙˙˙˙dŹ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:˙˙˙˙˙˙˙˙˙dŹ
 
_user_specified_nameinputs
§
ł
G__inference_sequential_layer_call_and_return_conditional_losses_3832186
dense_input!
dense_3832175:
ŹŹ
dense_3832177:	Ź#
dense_1_3832180:
ŹŹ
dense_1_3832182:	Ź
identity˘dense/StatefulPartitionedCall˘dense_1/StatefulPartitionedCallń
dense/StatefulPartitionedCallStatefulPartitionedCalldense_inputdense_3832175dense_3832177*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:˙˙˙˙˙˙˙˙˙dŹ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_3832031
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_3832180dense_1_3832182*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:˙˙˙˙˙˙˙˙˙dŹ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_3832067|
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙dŹ
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :˙˙˙˙˙˙˙˙˙dŹ: : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:Y U
,
_output_shapes
:˙˙˙˙˙˙˙˙˙dŹ
%
_user_specified_namedense_input
á
Ö
9__inference_emotion_detection_model_layer_call_fn_3833591
input_1
unknown:	dŹ
	unknown_0:
XŹ
	unknown_1:
ŹŹ
	unknown_2:
ŹŹ
	unknown_3:
ŹŹ
	unknown_4:	Ź
	unknown_5:	Ź
	unknown_6:
ŹŹ
	unknown_7:	Ź
	unknown_8:
ŹŹ
	unknown_9:	Ź

unknown_10:
ŹŹ

unknown_11:
ŹŹ

unknown_12:
ŹŹ

unknown_13:	Ź

unknown_14:	Ź

unknown_15:
ŹŹ

unknown_16:
ŹŹ

unknown_17:
ŹŹ

unknown_18:
ŹŹ

unknown_19:	Ź

unknown_20:
ŹŹ

unknown_21:	Ź

unknown_22:	Ź

unknown_23:
identity˘StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23*%
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*;
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_emotion_detection_model_layer_call_and_return_conditional_losses_3833483f
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:˙˙˙˙˙˙˙˙˙: : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:L H
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
!
_user_specified_name	input_1
ś
ü
B__inference_dense_layer_call_and_return_conditional_losses_3835179

inputs5
!tensordot_readvariableop_resource:
ŹŹ.
biasadd_readvariableop_resource:	Ź
identity˘BiasAdd/ReadVariableOp˘Tensordot/ReadVariableOp|
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource* 
_output_shapes
:
ŹŹ*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ť
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ż
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:z
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙dŹ
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ź\
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:ŹY
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : §
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙dŹs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:Ź*
dtype0}
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙dŹV
	LeakyRelu	LeakyReluBiasAdd:output:0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙dŹk
IdentityIdentityLeakyRelu:activations:0^NoOp*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙dŹz
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:˙˙˙˙˙˙˙˙˙dŹ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:˙˙˙˙˙˙˙˙˙dŹ
 
_user_specified_nameinputs
¸
ţ
D__inference_dense_2_layer_call_and_return_conditional_losses_3832224

inputs5
!tensordot_readvariableop_resource:
ŹŹ.
biasadd_readvariableop_resource:	Ź
identity˘BiasAdd/ReadVariableOp˘Tensordot/ReadVariableOp|
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource* 
_output_shapes
:
ŹŹ*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ť
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ż
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:z
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙dŹ
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ź\
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:ŹY
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : §
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙dŹs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:Ź*
dtype0}
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙dŹV
	LeakyRelu	LeakyReluBiasAdd:output:0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙dŹk
IdentityIdentityLeakyRelu:activations:0^NoOp*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙dŹz
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:˙˙˙˙˙˙˙˙˙dŹ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:˙˙˙˙˙˙˙˙˙dŹ
 
_user_specified_nameinputs


ö
D__inference_dense_4_layer_call_and_return_conditional_losses_3835139

inputs1
matmul_readvariableop_resource:	Ź-
biasadd_readvariableop_resource:
identity˘BiasAdd/ReadVariableOp˘MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	Ź*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:˙˙˙˙˙˙˙˙˙Ź: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ź
 
_user_specified_nameinputs
Íę
í
__inference_call_3831009
inputs_for_keys
inputs_for_values
inputs_for_queries5
!tensordot_readvariableop_resource:
ŹŹ7
#tensordot_1_readvariableop_resource:
ŹŹ7
#tensordot_2_readvariableop_resource:
ŹŹ
identity˘Tensordot/ReadVariableOp˘Tensordot_1/ReadVariableOp˘Tensordot_2/ReadVariableOp|
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource* 
_output_shapes
:
ŹŹ*
dtype0h
Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"'  ,  z
Tensordot/ReshapeReshapeinputs_for_keys Tensordot/Reshape/shape:output:0*
T0* 
_output_shapes
:
NŹ
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0* 
_output_shapes
:
NŹd
Tensordot/shapeConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ,  x
	TensordotReshapeTensordot/MatMul:product:0Tensordot/shape:output:0*
T0*#
_output_shapes
:ddŹ
Tensordot_1/ReadVariableOpReadVariableOp#tensordot_1_readvariableop_resource* 
_output_shapes
:
ŹŹ*
dtype0j
Tensordot_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"'  ,  
Tensordot_1/ReshapeReshapeinputs_for_values"Tensordot_1/Reshape/shape:output:0*
T0* 
_output_shapes
:
NŹ
Tensordot_1/MatMulMatMulTensordot_1/Reshape:output:0"Tensordot_1/ReadVariableOp:value:0*
T0* 
_output_shapes
:
NŹf
Tensordot_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ,  ~
Tensordot_1ReshapeTensordot_1/MatMul:product:0Tensordot_1/shape:output:0*
T0*#
_output_shapes
:ddŹ
Tensordot_2/ReadVariableOpReadVariableOp#tensordot_2_readvariableop_resource* 
_output_shapes
:
ŹŹ*
dtype0j
Tensordot_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"'  ,  
Tensordot_2/ReshapeReshapeinputs_for_queries"Tensordot_2/Reshape/shape:output:0*
T0* 
_output_shapes
:
NŹ
Tensordot_2/MatMulMatMulTensordot_2/Reshape:output:0"Tensordot_2/ReadVariableOp:value:0*
T0* 
_output_shapes
:
NŹf
Tensordot_2/shapeConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ,  ~
Tensordot_2ReshapeTensordot_2/MatMul:product:0Tensordot_2/shape:output:0*
T0*#
_output_shapes
:ddŹąš
attention_matrix_2/ConstConst*
_output_shapes

:dd*
dtype0*ß¸
valueÔ¸BĐ¸dd"Ŕ¸      ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙          ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙              ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                      ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                          ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                              ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                      ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                          ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                              ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                      ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                          ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                              ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                      ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                          ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                              ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                      ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                          ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                              ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                      ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                          ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                              ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                      ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                          ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                              ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                      ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                          ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                              ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                      ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                          ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                              ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                      ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                          ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                              ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                      ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                          ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                              ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                      ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                          ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                              ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                      ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                          ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                              ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                      ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                          ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                              ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                      ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                          ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                              ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                      ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                          ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                              ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                      ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                          ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                              ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                                  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                                      ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                                          ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                                              ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                                                  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                                                      ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                                                          ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                                                              ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                                                                  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                                                                      ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                                                                          ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                                                                              ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                                                                                  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                                                                                      ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                                                                                          ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                                                                                              ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                                                                                                  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                                                                                                      ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                                                                                                          ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                                                                                                              ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                                                                                                                  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                                                                                                                      ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                                                                                                                          ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                                                                                                                              ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                                                                                                                                  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                                                                                                                                      ˙  ˙  ˙                                                                                                                                                                                                                                                                                                                                                                                                          ˙  ˙                                                                                                                                                                                                                                                                                                                                                                                                              ˙                                                                                                                                                                                                                                                                                                                                                                                                                u
 attention_matrix_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"˙˙˙˙d   d    
attention_matrix_2/ReshapeReshape!attention_matrix_2/Const:output:0)attention_matrix_2/Reshape/shape:output:0*
T0*"
_output_shapes
:ddm
attention_matrix_2/ShapeConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ,  p
&attention_matrix_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(attention_matrix_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(attention_matrix_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:°
 attention_matrix_2/strided_sliceStridedSlice!attention_matrix_2/Shape:output:0/attention_matrix_2/strided_slice/stack:output:01attention_matrix_2/strided_slice/stack_1:output:01attention_matrix_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
#attention_matrix_2/Tile/multiples/1Const*
_output_shapes
: *
dtype0*
value	B :e
#attention_matrix_2/Tile/multiples/2Const*
_output_shapes
: *
dtype0*
value	B :Ţ
!attention_matrix_2/Tile/multiplesPack)attention_matrix_2/strided_slice:output:0,attention_matrix_2/Tile/multiples/1:output:0,attention_matrix_2/Tile/multiples/2:output:0*
N*
T0*
_output_shapes
:
attention_matrix_2/TileTile#attention_matrix_2/Reshape:output:0*attention_matrix_2/Tile/multiples:output:0*
T0*"
_output_shapes
:dddv
!attention_matrix_2/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
attention_matrix_2/transpose	TransposeTensordot:output:0*attention_matrix_2/transpose/perm:output:0*
T0*#
_output_shapes
:dŹd
attention_matrix_2/MatMulBatchMatMulV2Tensordot_2:output:0 attention_matrix_2/transpose:y:0*
T0*"
_output_shapes
:ddd[
attention_matrix_2/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :ds
attention_matrix_2/CastCast"attention_matrix_2/Cast/x:output:0*

DstT0*

SrcT0*
_output_shapes
: ]
attention_matrix_2/SqrtSqrtattention_matrix_2/Cast:y:0*
T0*
_output_shapes
: 
attention_matrix_2/truedivRealDiv"attention_matrix_2/MatMul:output:0attention_matrix_2/Sqrt:y:0*
T0*"
_output_shapes
:ddd
attention_matrix_2/addAddV2attention_matrix_2/truediv:z:0 attention_matrix_2/Tile:output:0*
T0*"
_output_shapes
:dddn
attention_matrix_2/SoftmaxSoftmaxattention_matrix_2/add:z:0*
T0*"
_output_shapes
:ddd
MatMulBatchMatMulV2$attention_matrix_2/Softmax:softmax:0Tensordot_1:output:0*
T0*#
_output_shapes
:ddŹZ
IdentityIdentityMatMul:output:0^NoOp*
T0*#
_output_shapes
:ddŹ
NoOpNoOp^Tensordot/ReadVariableOp^Tensordot_1/ReadVariableOp^Tensordot_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:ddŹ:ddŹ:ddŹ: : : 24
Tensordot/ReadVariableOpTensordot/ReadVariableOp28
Tensordot_1/ReadVariableOpTensordot_1/ReadVariableOp28
Tensordot_2/ReadVariableOpTensordot_2/ReadVariableOp:T P
#
_output_shapes
:ddŹ
)
_user_specified_nameinputs_for_keys:VR
#
_output_shapes
:ddŹ
+
_user_specified_nameinputs_for_values:WS
#
_output_shapes
:ddŹ
,
_user_specified_nameinputs_for_queries

Ž
G__inference_sequential_layer_call_and_return_conditional_losses_3832074

inputs!
dense_3832032:
ŹŹ
dense_3832034:	Ź#
dense_1_3832068:
ŹŹ
dense_1_3832070:	Ź
identity˘dense/StatefulPartitionedCall˘dense_1/StatefulPartitionedCallě
dense/StatefulPartitionedCallStatefulPartitionedCallinputsdense_3832032dense_3832034*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:˙˙˙˙˙˙˙˙˙dŹ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_3832031
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_3832068dense_1_3832070*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:˙˙˙˙˙˙˙˙˙dŹ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_3832067|
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙dŹ
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :˙˙˙˙˙˙˙˙˙dŹ: : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:T P
,
_output_shapes
:˙˙˙˙˙˙˙˙˙dŹ
 
_user_specified_nameinputs
Üĺ
í
__inference_call_3831053
inputs_for_keys
inputs_for_values
inputs_for_queries5
!tensordot_readvariableop_resource:
ŹŹ7
#tensordot_1_readvariableop_resource:
ŹŹ7
#tensordot_2_readvariableop_resource:
ŹŹ
identity˘Tensordot/ReadVariableOp˘Tensordot_1/ReadVariableOp˘Tensordot_2/ReadVariableOp|
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource* 
_output_shapes
:
ŹŹ*
dtype0h
Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"'  ,  z
Tensordot/ReshapeReshapeinputs_for_keys Tensordot/Reshape/shape:output:0*
T0* 
_output_shapes
:
NŹ
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0* 
_output_shapes
:
NŹd
Tensordot/shapeConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ,  x
	TensordotReshapeTensordot/MatMul:product:0Tensordot/shape:output:0*
T0*#
_output_shapes
:ddŹ
Tensordot_1/ReadVariableOpReadVariableOp#tensordot_1_readvariableop_resource* 
_output_shapes
:
ŹŹ*
dtype0j
Tensordot_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"'  ,  
Tensordot_1/ReshapeReshapeinputs_for_values"Tensordot_1/Reshape/shape:output:0*
T0* 
_output_shapes
:
NŹ
Tensordot_1/MatMulMatMulTensordot_1/Reshape:output:0"Tensordot_1/ReadVariableOp:value:0*
T0* 
_output_shapes
:
NŹf
Tensordot_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ,  ~
Tensordot_1ReshapeTensordot_1/MatMul:product:0Tensordot_1/shape:output:0*
T0*#
_output_shapes
:ddŹ
Tensordot_2/ReadVariableOpReadVariableOp#tensordot_2_readvariableop_resource* 
_output_shapes
:
ŹŹ*
dtype0~
Tensordot_2/MatMulMatMulinputs_for_queries"Tensordot_2/ReadVariableOp:value:0*
T0*
_output_shapes
:	dŹąš
attention_matrix_3/ConstConst*
_output_shapes

:dd*
dtype0*ß¸
valueÔ¸BĐ¸dd"Ŕ¸      ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙          ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙              ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                      ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                          ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                              ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                      ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                          ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                              ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                      ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                          ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                              ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                      ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                          ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                              ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                      ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                          ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                              ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                      ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                          ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                              ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                      ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                          ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                              ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                      ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                          ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                              ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                      ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                          ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                              ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                      ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                          ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                              ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                      ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                          ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                              ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                      ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                          ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                              ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                      ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                          ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                              ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                      ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                          ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                              ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                      ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                          ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                              ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                      ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                          ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                              ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                      ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                          ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                              ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                                  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                                      ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                                          ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                                              ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                                                  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                                                      ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                                                          ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                                                              ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                                                                  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                                                                      ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                                                                          ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                                                                              ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                                                                                  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                                                                                      ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                                                                                          ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                                                                                              ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                                                                                                  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                                                                                                      ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                                                                                                          ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                                                                                                              ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                                                                                                                  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                                                                                                                      ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                                                                                                                          ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                                                                                                                              ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                                                                                                                                  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                                                                                                                                      ˙  ˙  ˙                                                                                                                                                                                                                                                                                                                                                                                                          ˙  ˙                                                                                                                                                                                                                                                                                                                                                                                                              ˙                                                                                                                                                                                                                                                                                                                                                                                                                u
 attention_matrix_3/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"˙˙˙˙d   d    
attention_matrix_3/ReshapeReshape!attention_matrix_3/Const:output:0)attention_matrix_3/Reshape/shape:output:0*
T0*"
_output_shapes
:ddm
attention_matrix_3/ShapeConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ,  p
&attention_matrix_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(attention_matrix_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(attention_matrix_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:°
 attention_matrix_3/strided_sliceStridedSlice!attention_matrix_3/Shape:output:0/attention_matrix_3/strided_slice/stack:output:01attention_matrix_3/strided_slice/stack_1:output:01attention_matrix_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
#attention_matrix_3/Tile/multiples/1Const*
_output_shapes
: *
dtype0*
value	B :e
#attention_matrix_3/Tile/multiples/2Const*
_output_shapes
: *
dtype0*
value	B :Ţ
!attention_matrix_3/Tile/multiplesPack)attention_matrix_3/strided_slice:output:0,attention_matrix_3/Tile/multiples/1:output:0,attention_matrix_3/Tile/multiples/2:output:0*
N*
T0*
_output_shapes
:
attention_matrix_3/TileTile#attention_matrix_3/Reshape:output:0*attention_matrix_3/Tile/multiples:output:0*
T0*"
_output_shapes
:dddv
!attention_matrix_3/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
attention_matrix_3/transpose	TransposeTensordot:output:0*attention_matrix_3/transpose/perm:output:0*
T0*#
_output_shapes
:dŹd
attention_matrix_3/MatMulBatchMatMulV2Tensordot_2/MatMul:product:0 attention_matrix_3/transpose:y:0*
T0*"
_output_shapes
:ddd[
attention_matrix_3/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :ds
attention_matrix_3/CastCast"attention_matrix_3/Cast/x:output:0*

DstT0*

SrcT0*
_output_shapes
: ]
attention_matrix_3/SqrtSqrtattention_matrix_3/Cast:y:0*
T0*
_output_shapes
: 
attention_matrix_3/truedivRealDiv"attention_matrix_3/MatMul:output:0attention_matrix_3/Sqrt:y:0*
T0*"
_output_shapes
:dddr
attention_matrix_3/SoftmaxSoftmaxattention_matrix_3/truediv:z:0*
T0*"
_output_shapes
:ddd
MatMulBatchMatMulV2$attention_matrix_3/Softmax:softmax:0Tensordot_1:output:0*
T0*#
_output_shapes
:ddŹZ
IdentityIdentityMatMul:output:0^NoOp*
T0*#
_output_shapes
:ddŹ
NoOpNoOp^Tensordot/ReadVariableOp^Tensordot_1/ReadVariableOp^Tensordot_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:ddŹ:ddŹ:	dŹ: : : 24
Tensordot/ReadVariableOpTensordot/ReadVariableOp28
Tensordot_1/ReadVariableOpTensordot_1/ReadVariableOp28
Tensordot_2/ReadVariableOpTensordot_2/ReadVariableOp:T P
#
_output_shapes
:ddŹ
)
_user_specified_nameinputs_for_keys:VR
#
_output_shapes
:ddŹ
+
_user_specified_nameinputs_for_values:SO

_output_shapes
:	dŹ
,
_user_specified_nameinputs_for_queries
čV
×
"__inference__wrapped_model_3831993
input_1l
Yemotion_detection_model_token_and_position_embedding_embedding_1_embedding_lookup_3831925:	dŹk
Wemotion_detection_model_token_and_position_embedding_embedding_embedding_lookup_3831930:
XŹE
1emotion_detection_model_transformer_block_3831936:
ŹŹE
1emotion_detection_model_transformer_block_3831938:
ŹŹE
1emotion_detection_model_transformer_block_3831940:
ŹŹ@
1emotion_detection_model_transformer_block_3831942:	Ź@
1emotion_detection_model_transformer_block_3831944:	ŹE
1emotion_detection_model_transformer_block_3831946:
ŹŹ@
1emotion_detection_model_transformer_block_3831948:	ŹE
1emotion_detection_model_transformer_block_3831950:
ŹŹ@
1emotion_detection_model_transformer_block_3831952:	ŹG
3emotion_detection_model_transformer_block_1_3831956:
ŹŹG
3emotion_detection_model_transformer_block_1_3831958:
ŹŹG
3emotion_detection_model_transformer_block_1_3831960:
ŹŹB
3emotion_detection_model_transformer_block_1_3831962:	ŹB
3emotion_detection_model_transformer_block_1_3831964:	ŹG
3emotion_detection_model_transformer_block_1_3831966:
ŹŹG
3emotion_detection_model_transformer_block_1_3831968:
ŹŹG
3emotion_detection_model_transformer_block_1_3831970:
ŹŹG
3emotion_detection_model_transformer_block_1_3831972:
ŹŹB
3emotion_detection_model_transformer_block_1_3831974:	ŹG
3emotion_detection_model_transformer_block_1_3831976:
ŹŹB
3emotion_detection_model_transformer_block_1_3831978:	Ź^
Kemotion_detection_model_sequential_2_dense_4_matmul_readvariableop_resource:	ŹZ
Lemotion_detection_model_sequential_2_dense_4_biasadd_readvariableop_resource:
identity˘Cemotion_detection_model/sequential_2/dense_4/BiasAdd/ReadVariableOp˘Bemotion_detection_model/sequential_2/dense_4/MatMul/ReadVariableOp˘Oemotion_detection_model/token_and_position_embedding/embedding/embedding_lookup˘Qemotion_detection_model/token_and_position_embedding/embedding_1/embedding_lookup˘Aemotion_detection_model/transformer_block/StatefulPartitionedCall˘Cemotion_detection_model/transformer_block_1/StatefulPartitionedCall
@emotion_detection_model/token_and_position_embedding/range/startConst*
_output_shapes
: *
dtype0*
value	B : 
@emotion_detection_model/token_and_position_embedding/range/limitConst*
_output_shapes
: *
dtype0*
value	B :d
@emotion_detection_model/token_and_position_embedding/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :Ŕ
:emotion_detection_model/token_and_position_embedding/rangeRangeIemotion_detection_model/token_and_position_embedding/range/start:output:0Iemotion_detection_model/token_and_position_embedding/range/limit:output:0Iemotion_detection_model/token_and_position_embedding/range/delta:output:0*
_output_shapes
:dŻ
Qemotion_detection_model/token_and_position_embedding/embedding_1/embedding_lookupResourceGatherYemotion_detection_model_token_and_position_embedding_embedding_1_embedding_lookup_3831925Cemotion_detection_model/token_and_position_embedding/range:output:0*
Tindices0*l
_classb
`^loc:@emotion_detection_model/token_and_position_embedding/embedding_1/embedding_lookup/3831925*
_output_shapes
:	dŹ*
dtype0Ú
Zemotion_detection_model/token_and_position_embedding/embedding_1/embedding_lookup/IdentityIdentityZemotion_detection_model/token_and_position_embedding/embedding_1/embedding_lookup:output:0*
T0*l
_classb
`^loc:@emotion_detection_model/token_and_position_embedding/embedding_1/embedding_lookup/3831925*
_output_shapes
:	dŹ÷
\emotion_detection_model/token_and_position_embedding/embedding_1/embedding_lookup/Identity_1Identitycemotion_detection_model/token_and_position_embedding/embedding_1/embedding_lookup/Identity:output:0*
T0*
_output_shapes
:	dŹö
Oemotion_detection_model/token_and_position_embedding/embedding/embedding_lookupResourceGatherWemotion_detection_model_token_and_position_embedding_embedding_embedding_lookup_3831930input_1*
Tindices0*j
_class`
^\loc:@emotion_detection_model/token_and_position_embedding/embedding/embedding_lookup/3831930*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ź*
dtype0Ý
Xemotion_detection_model/token_and_position_embedding/embedding/embedding_lookup/IdentityIdentityXemotion_detection_model/token_and_position_embedding/embedding/embedding_lookup:output:0*
T0*j
_class`
^\loc:@emotion_detection_model/token_and_position_embedding/embedding/embedding_lookup/3831930*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Źü
Zemotion_detection_model/token_and_position_embedding/embedding/embedding_lookup/Identity_1Identityaemotion_detection_model/token_and_position_embedding/embedding/embedding_lookup/Identity:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Źˇ
8emotion_detection_model/token_and_position_embedding/addAddV2cemotion_detection_model/token_and_position_embedding/embedding/embedding_lookup/Identity_1:output:0eemotion_detection_model/token_and_position_embedding/embedding_1/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:	dŹÎ
Aemotion_detection_model/transformer_block/StatefulPartitionedCallStatefulPartitionedCall<emotion_detection_model/token_and_position_embedding/add:z:01emotion_detection_model_transformer_block_38319361emotion_detection_model_transformer_block_38319381emotion_detection_model_transformer_block_38319401emotion_detection_model_transformer_block_38319421emotion_detection_model_transformer_block_38319441emotion_detection_model_transformer_block_38319461emotion_detection_model_transformer_block_38319481emotion_detection_model_transformer_block_38319501emotion_detection_model_transformer_block_3831952*
Tin
2
*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:ddŹ*+
_read_only_resource_inputs
		*-
config_proto

CPU

GPU 2J 8 *!
fR
__inference_call_3827515Ž
(emotion_detection_model/dropout/IdentityIdentityJemotion_detection_model/transformer_block/StatefulPartitionedCall:output:0*
T0*#
_output_shapes
:ddŹť
Cemotion_detection_model/transformer_block_1/StatefulPartitionedCallStatefulPartitionedCall1emotion_detection_model/dropout/Identity:output:0<emotion_detection_model/token_and_position_embedding/add:z:03emotion_detection_model_transformer_block_1_38319563emotion_detection_model_transformer_block_1_38319583emotion_detection_model_transformer_block_1_38319603emotion_detection_model_transformer_block_1_38319623emotion_detection_model_transformer_block_1_38319643emotion_detection_model_transformer_block_1_38319663emotion_detection_model_transformer_block_1_38319683emotion_detection_model_transformer_block_1_38319703emotion_detection_model_transformer_block_1_38319723emotion_detection_model_transformer_block_1_38319743emotion_detection_model_transformer_block_1_38319763emotion_detection_model_transformer_block_1_3831978*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:ddŹ*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *!
fR
__inference_call_3827796h
&emotion_detection_model/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : á
"emotion_detection_model/ExpandDims
ExpandDimsLemotion_detection_model/transformer_block_1/StatefulPartitionedCall:output:0/emotion_detection_model/ExpandDims/dim:output:0*
T0*'
_output_shapes
:ddŹ
Bemotion_detection_model/global_max_pooling2d/Max/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      Ű
0emotion_detection_model/global_max_pooling2d/MaxMax+emotion_detection_model/ExpandDims:output:0Kemotion_detection_model/global_max_pooling2d/Max/reduction_indices:output:0*
T0*
_output_shapes
:	ŹĎ
Bemotion_detection_model/sequential_2/dense_4/MatMul/ReadVariableOpReadVariableOpKemotion_detection_model_sequential_2_dense_4_matmul_readvariableop_resource*
_output_shapes
:	Ź*
dtype0í
3emotion_detection_model/sequential_2/dense_4/MatMulMatMul9emotion_detection_model/global_max_pooling2d/Max:output:0Jemotion_detection_model/sequential_2/dense_4/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:Ě
Cemotion_detection_model/sequential_2/dense_4/BiasAdd/ReadVariableOpReadVariableOpLemotion_detection_model_sequential_2_dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ô
4emotion_detection_model/sequential_2/dense_4/BiasAddBiasAdd=emotion_detection_model/sequential_2/dense_4/MatMul:product:0Kemotion_detection_model/sequential_2/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:§
4emotion_detection_model/sequential_2/dense_4/SigmoidSigmoid=emotion_detection_model/sequential_2/dense_4/BiasAdd:output:0*
T0*
_output_shapes

:~
IdentityIdentity8emotion_detection_model/sequential_2/dense_4/Sigmoid:y:0^NoOp*
T0*
_output_shapes

:
NoOpNoOpD^emotion_detection_model/sequential_2/dense_4/BiasAdd/ReadVariableOpC^emotion_detection_model/sequential_2/dense_4/MatMul/ReadVariableOpP^emotion_detection_model/token_and_position_embedding/embedding/embedding_lookupR^emotion_detection_model/token_and_position_embedding/embedding_1/embedding_lookupB^emotion_detection_model/transformer_block/StatefulPartitionedCallD^emotion_detection_model/transformer_block_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:˙˙˙˙˙˙˙˙˙: : : : : : : : : : : : : : : : : : : : : : : : : 2
Cemotion_detection_model/sequential_2/dense_4/BiasAdd/ReadVariableOpCemotion_detection_model/sequential_2/dense_4/BiasAdd/ReadVariableOp2
Bemotion_detection_model/sequential_2/dense_4/MatMul/ReadVariableOpBemotion_detection_model/sequential_2/dense_4/MatMul/ReadVariableOp2˘
Oemotion_detection_model/token_and_position_embedding/embedding/embedding_lookupOemotion_detection_model/token_and_position_embedding/embedding/embedding_lookup2Ś
Qemotion_detection_model/token_and_position_embedding/embedding_1/embedding_lookupQemotion_detection_model/token_and_position_embedding/embedding_1/embedding_lookup2
Aemotion_detection_model/transformer_block/StatefulPartitionedCallAemotion_detection_model/transformer_block/StatefulPartitionedCall2
Cemotion_detection_model/transformer_block_1/StatefulPartitionedCallCemotion_detection_model/transformer_block_1/StatefulPartitionedCall:L H
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
!
_user_specified_name	input_1
Ů
ţ
D__inference_dense_3_layer_call_and_return_conditional_losses_3832260

inputs5
!tensordot_readvariableop_resource:
ŹŹ.
biasadd_readvariableop_resource:	Ź
identity˘BiasAdd/ReadVariableOp˘Tensordot/ReadVariableOp|
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource* 
_output_shapes
:
ŹŹ*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ť
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ż
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:z
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙dŹ
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ź\
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:ŹY
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : §
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙dŹs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:Ź*
dtype0}
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙dŹd
IdentityIdentityBiasAdd:output:0^NoOp*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙dŹz
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:˙˙˙˙˙˙˙˙˙dŹ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:˙˙˙˙˙˙˙˙˙dŹ
 
_user_specified_nameinputs


+__inference_embedding_layer_call_fn_3834046

inputs
unknown:
XŹ
identity˘StatefulPartitionedCallĎ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ź*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_embedding_layer_call_and_return_conditional_losses_3832516p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ź`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:˙˙˙˙˙˙˙˙˙: 22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
ô	
í
3__inference_transformer_block_layer_call_fn_3834106

inputs
unknown:
ŹŹ
	unknown_0:
ŹŹ
	unknown_1:
ŹŹ
	unknown_2:	Ź
	unknown_3:	Ź
	unknown_4:
ŹŹ
	unknown_5:	Ź
	unknown_6:
ŹŹ
	unknown_7:	Ź
identity˘StatefulPartitionedCallş
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
Tin
2
*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:ddŹ*+
_read_only_resource_inputs
		*-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_transformer_block_layer_call_and_return_conditional_losses_3832646k
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*#
_output_shapes
:ddŹ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:	dŹ: : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:G C

_output_shapes
:	dŹ
 
_user_specified_nameinputs
Ĺ
Ú
,__inference_sequential_layer_call_fn_3832158
dense_input
unknown:
ŹŹ
	unknown_0:	Ź
	unknown_1:
ŹŹ
	unknown_2:	Ź
identity˘StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCalldense_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:˙˙˙˙˙˙˙˙˙dŹ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_sequential_layer_call_and_return_conditional_losses_3832134t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙dŹ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :˙˙˙˙˙˙˙˙˙dŹ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
,
_output_shapes
:˙˙˙˙˙˙˙˙˙dŹ
%
_user_specified_namedense_input
Ĺ
Ú
,__inference_sequential_layer_call_fn_3832085
dense_input
unknown:
ŹŹ
	unknown_0:	Ź
	unknown_1:
ŹŹ
	unknown_2:	Ź
identity˘StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCalldense_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:˙˙˙˙˙˙˙˙˙dŹ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_sequential_layer_call_and_return_conditional_losses_3832074t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙dŹ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :˙˙˙˙˙˙˙˙˙dŹ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
,
_output_shapes
:˙˙˙˙˙˙˙˙˙dŹ
%
_user_specified_namedense_input
á
Ö
9__inference_emotion_detection_model_layer_call_fn_3832929
input_1
unknown:	dŹ
	unknown_0:
XŹ
	unknown_1:
ŹŹ
	unknown_2:
ŹŹ
	unknown_3:
ŹŹ
	unknown_4:	Ź
	unknown_5:	Ź
	unknown_6:
ŹŹ
	unknown_7:	Ź
	unknown_8:
ŹŹ
	unknown_9:	Ź

unknown_10:
ŹŹ

unknown_11:
ŹŹ

unknown_12:
ŹŹ

unknown_13:	Ź

unknown_14:	Ź

unknown_15:
ŹŹ

unknown_16:
ŹŹ

unknown_17:
ŹŹ

unknown_18:
ŹŹ

unknown_19:	Ź

unknown_20:
ŹŹ

unknown_21:	Ź

unknown_22:	Ź

unknown_23:
identity˘StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23*%
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*;
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_emotion_detection_model_layer_call_and_return_conditional_losses_3832876f
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:˙˙˙˙˙˙˙˙˙: : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:L H
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
!
_user_specified_name	input_1
ÄŢ
í
__inference_call_3830564
inputs_for_keys
inputs_for_values
inputs_for_queries5
!tensordot_readvariableop_resource:
ŹŹ7
#tensordot_1_readvariableop_resource:
ŹŹ7
#tensordot_2_readvariableop_resource:
ŹŹ
identity˘Tensordot/ReadVariableOp˘Tensordot_1/ReadVariableOp˘Tensordot_2/ReadVariableOp|
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource* 
_output_shapes
:
ŹŹ*
dtype0w
Tensordot/MatMulMatMulinputs_for_keys Tensordot/ReadVariableOp:value:0*
T0*
_output_shapes
:	dŹ
Tensordot_1/ReadVariableOpReadVariableOp#tensordot_1_readvariableop_resource* 
_output_shapes
:
ŹŹ*
dtype0}
Tensordot_1/MatMulMatMulinputs_for_values"Tensordot_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	dŹ
Tensordot_2/ReadVariableOpReadVariableOp#tensordot_2_readvariableop_resource* 
_output_shapes
:
ŹŹ*
dtype0~
Tensordot_2/MatMulMatMulinputs_for_queries"Tensordot_2/ReadVariableOp:value:0*
T0*
_output_shapes
:	dŹŻš
attention_matrix/ConstConst*
_output_shapes

:dd*
dtype0*ß¸
valueÔ¸BĐ¸dd"Ŕ¸      ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙          ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙              ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                      ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                          ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                              ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                      ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                          ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                              ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                      ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                          ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                              ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                      ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                          ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                              ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                      ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                          ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                              ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                      ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                          ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                              ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                      ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                          ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                              ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                      ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                          ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                              ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                      ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                          ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                              ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                      ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                          ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                              ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                      ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                          ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                              ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                      ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                          ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                              ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                      ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                          ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                              ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                      ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                          ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                              ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                      ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                          ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                              ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                      ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                          ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                              ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                      ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                          ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                              ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                                  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                                      ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                                          ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                                              ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                                                  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                                                      ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                                                          ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                                                              ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                                                                  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                                                                      ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                                                                          ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                                                                              ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                                                                                  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                                                                                      ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                                                                                          ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                                                                                              ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                                                                                                  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                                                                                                      ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                                                                                                          ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                                                                                                              ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                                                                                                                  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                                                                                                                      ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                                                                                                                          ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                                                                                                                              ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                                                                                                                                  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                                                                                                                                      ˙  ˙  ˙                                                                                                                                                                                                                                                                                                                                                                                                          ˙  ˙                                                                                                                                                                                                                                                                                                                                                                                                              ˙                                                                                                                                                                                                                                                                                                                                                                                                                s
attention_matrix/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"˙˙˙˙d   d   
attention_matrix/ReshapeReshapeattention_matrix/Const:output:0'attention_matrix/Reshape/shape:output:0*
T0*"
_output_shapes
:ddg
attention_matrix/ShapeConst*
_output_shapes
:*
dtype0*
valueB"d   ,  n
$attention_matrix/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&attention_matrix/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&attention_matrix/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ś
attention_matrix/strided_sliceStridedSliceattention_matrix/Shape:output:0-attention_matrix/strided_slice/stack:output:0/attention_matrix/strided_slice/stack_1:output:0/attention_matrix/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskc
!attention_matrix/Tile/multiples/1Const*
_output_shapes
: *
dtype0*
value	B :c
!attention_matrix/Tile/multiples/2Const*
_output_shapes
: *
dtype0*
value	B :Ö
attention_matrix/Tile/multiplesPack'attention_matrix/strided_slice:output:0*attention_matrix/Tile/multiples/1:output:0*attention_matrix/Tile/multiples/2:output:0*
N*
T0*
_output_shapes
:
attention_matrix/TileTile!attention_matrix/Reshape:output:0(attention_matrix/Tile/multiples:output:0*
T0*"
_output_shapes
:dddp
attention_matrix/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       
attention_matrix/transpose	TransposeTensordot/MatMul:product:0(attention_matrix/transpose/perm:output:0*
T0*
_output_shapes
:	Źd
attention_matrix/MatMulMatMulTensordot_2/MatMul:product:0attention_matrix/transpose:y:0*
T0*
_output_shapes

:ddY
attention_matrix/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :do
attention_matrix/CastCast attention_matrix/Cast/x:output:0*

DstT0*

SrcT0*
_output_shapes
: Y
attention_matrix/SqrtSqrtattention_matrix/Cast:y:0*
T0*
_output_shapes
: 
attention_matrix/truedivRealDiv!attention_matrix/MatMul:product:0attention_matrix/Sqrt:y:0*
T0*
_output_shapes

:dd
attention_matrix/addAddV2attention_matrix/truediv:z:0attention_matrix/Tile:output:0*
T0*"
_output_shapes
:dddj
attention_matrix/SoftmaxSoftmaxattention_matrix/add:z:0*
T0*"
_output_shapes
:ddd
MatMulBatchMatMulV2"attention_matrix/Softmax:softmax:0Tensordot_1/MatMul:product:0*
T0*#
_output_shapes
:ddŹZ
IdentityIdentityMatMul:output:0^NoOp*
T0*#
_output_shapes
:ddŹ
NoOpNoOp^Tensordot/ReadVariableOp^Tensordot_1/ReadVariableOp^Tensordot_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':	dŹ:	dŹ:	dŹ: : : 24
Tensordot/ReadVariableOpTensordot/ReadVariableOp28
Tensordot_1/ReadVariableOpTensordot_1/ReadVariableOp28
Tensordot_2/ReadVariableOpTensordot_2/ReadVariableOp:P L

_output_shapes
:	dŹ
)
_user_specified_nameinputs_for_keys:RN

_output_shapes
:	dŹ
+
_user_specified_nameinputs_for_values:SO

_output_shapes
:	dŹ
,
_user_specified_nameinputs_for_queries
­
m
Q__inference_global_max_pooling2d_layer_call_and_return_conditional_losses_3834772

inputs
identityf
Max/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      m
MaxMaxinputsMax/reduction_indices:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙]
IdentityIdentityMax:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:r n
J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Ć
Ô
I__inference_sequential_2_layer_call_and_return_conditional_losses_3832488
dense_4_input"
dense_4_3832482:	Ź
dense_4_3832484:
identity˘dense_4/StatefulPartitionedCallö
dense_4/StatefulPartitionedCallStatefulPartitionedCalldense_4_inputdense_4_3832482dense_4_3832484*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_4_layer_call_and_return_conditional_losses_3832410w
IdentityIdentity(dense_4/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙h
NoOpNoOp ^dense_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:˙˙˙˙˙˙˙˙˙Ź: : 2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall:W S
(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ź
'
_user_specified_namedense_4_input
Ď

.__inference_sequential_2_layer_call_fn_3834781

inputs
unknown:	Ź
	unknown_0:
identity˘StatefulPartitionedCallŢ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sequential_2_layer_call_and_return_conditional_losses_3832417o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:˙˙˙˙˙˙˙˙˙Ź: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ź
 
_user_specified_nameinputs

ă
5__inference_transformer_block_1_layer_call_fn_3834395

inputs
context_sequence
unknown:
ŹŹ
	unknown_0:
ŹŹ
	unknown_1:
ŹŹ
	unknown_2:	Ź
	unknown_3:	Ź
	unknown_4:
ŹŹ
	unknown_5:
ŹŹ
	unknown_6:
ŹŹ
	unknown_7:
ŹŹ
	unknown_8:	Ź
	unknown_9:
ŹŹ

unknown_10:	Ź
identity˘StatefulPartitionedCall÷
StatefulPartitionedCallStatefulPartitionedCallinputscontext_sequenceunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:ddŹ*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_transformer_block_1_layer_call_and_return_conditional_losses_3832841k
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*#
_output_shapes
:ddŹ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:ddŹ:	dŹ: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:ddŹ
 
_user_specified_nameinputs:QM

_output_shapes
:	dŹ
*
_user_specified_namecontext_sequence
Ć
Ô
I__inference_sequential_2_layer_call_and_return_conditional_losses_3832479
dense_4_input"
dense_4_3832473:	Ź
dense_4_3832475:
identity˘dense_4/StatefulPartitionedCallö
dense_4/StatefulPartitionedCallStatefulPartitionedCalldense_4_inputdense_4_3832473dense_4_3832475*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_4_layer_call_and_return_conditional_losses_3832410w
IdentityIdentity(dense_4/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙h
NoOpNoOp ^dense_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:˙˙˙˙˙˙˙˙˙Ź: : 2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall:W S
(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ź
'
_user_specified_namedense_4_input
ÄŢ
í
__inference_call_3827399
inputs_for_keys
inputs_for_values
inputs_for_queries5
!tensordot_readvariableop_resource:
ŹŹ7
#tensordot_1_readvariableop_resource:
ŹŹ7
#tensordot_2_readvariableop_resource:
ŹŹ
identity˘Tensordot/ReadVariableOp˘Tensordot_1/ReadVariableOp˘Tensordot_2/ReadVariableOp|
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource* 
_output_shapes
:
ŹŹ*
dtype0w
Tensordot/MatMulMatMulinputs_for_keys Tensordot/ReadVariableOp:value:0*
T0*
_output_shapes
:	dŹ
Tensordot_1/ReadVariableOpReadVariableOp#tensordot_1_readvariableop_resource* 
_output_shapes
:
ŹŹ*
dtype0}
Tensordot_1/MatMulMatMulinputs_for_values"Tensordot_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	dŹ
Tensordot_2/ReadVariableOpReadVariableOp#tensordot_2_readvariableop_resource* 
_output_shapes
:
ŹŹ*
dtype0~
Tensordot_2/MatMulMatMulinputs_for_queries"Tensordot_2/ReadVariableOp:value:0*
T0*
_output_shapes
:	dŹŻš
attention_matrix/ConstConst*
_output_shapes

:dd*
dtype0*ß¸
valueÔ¸BĐ¸dd"Ŕ¸      ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙          ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙              ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                      ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                          ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                              ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                      ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                          ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                              ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                      ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                          ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                              ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                      ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                          ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                              ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                      ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                          ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                              ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                      ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                          ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                              ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                      ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                          ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                              ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                      ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                          ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                              ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                      ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                          ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                              ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                      ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                          ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                              ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                      ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                          ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                              ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                      ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                          ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                              ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                      ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                          ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                              ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                      ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                          ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                              ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                      ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                          ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                              ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                      ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                          ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                              ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                      ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                          ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                              ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                                  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                                      ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                                          ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                                              ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                                                  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                                                      ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                                                          ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                                                              ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                                                                  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                                                                      ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                                                                          ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                                                                              ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                                                                                  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                                                                                      ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                                                                                          ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                                                                                              ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                                                                                                  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                                                                                                      ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                                                                                                          ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                                                                                                              ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                                                                                                                  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                                                                                                                      ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                                                                                                                          ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                                                                                                                              ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                                                                                                                                  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                                                                                                                                      ˙  ˙  ˙                                                                                                                                                                                                                                                                                                                                                                                                          ˙  ˙                                                                                                                                                                                                                                                                                                                                                                                                              ˙                                                                                                                                                                                                                                                                                                                                                                                                                s
attention_matrix/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"˙˙˙˙d   d   
attention_matrix/ReshapeReshapeattention_matrix/Const:output:0'attention_matrix/Reshape/shape:output:0*
T0*"
_output_shapes
:ddg
attention_matrix/ShapeConst*
_output_shapes
:*
dtype0*
valueB"d   ,  n
$attention_matrix/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&attention_matrix/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&attention_matrix/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ś
attention_matrix/strided_sliceStridedSliceattention_matrix/Shape:output:0-attention_matrix/strided_slice/stack:output:0/attention_matrix/strided_slice/stack_1:output:0/attention_matrix/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskc
!attention_matrix/Tile/multiples/1Const*
_output_shapes
: *
dtype0*
value	B :c
!attention_matrix/Tile/multiples/2Const*
_output_shapes
: *
dtype0*
value	B :Ö
attention_matrix/Tile/multiplesPack'attention_matrix/strided_slice:output:0*attention_matrix/Tile/multiples/1:output:0*attention_matrix/Tile/multiples/2:output:0*
N*
T0*
_output_shapes
:
attention_matrix/TileTile!attention_matrix/Reshape:output:0(attention_matrix/Tile/multiples:output:0*
T0*"
_output_shapes
:dddp
attention_matrix/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       
attention_matrix/transpose	TransposeTensordot/MatMul:product:0(attention_matrix/transpose/perm:output:0*
T0*
_output_shapes
:	Źd
attention_matrix/MatMulMatMulTensordot_2/MatMul:product:0attention_matrix/transpose:y:0*
T0*
_output_shapes

:ddY
attention_matrix/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :do
attention_matrix/CastCast attention_matrix/Cast/x:output:0*

DstT0*

SrcT0*
_output_shapes
: Y
attention_matrix/SqrtSqrtattention_matrix/Cast:y:0*
T0*
_output_shapes
: 
attention_matrix/truedivRealDiv!attention_matrix/MatMul:product:0attention_matrix/Sqrt:y:0*
T0*
_output_shapes

:dd
attention_matrix/addAddV2attention_matrix/truediv:z:0attention_matrix/Tile:output:0*
T0*"
_output_shapes
:dddj
attention_matrix/SoftmaxSoftmaxattention_matrix/add:z:0*
T0*"
_output_shapes
:ddd
MatMulBatchMatMulV2"attention_matrix/Softmax:softmax:0Tensordot_1/MatMul:product:0*
T0*#
_output_shapes
:ddŹZ
IdentityIdentityMatMul:output:0^NoOp*
T0*#
_output_shapes
:ddŹ
NoOpNoOp^Tensordot/ReadVariableOp^Tensordot_1/ReadVariableOp^Tensordot_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':	dŹ:	dŹ:	dŹ: : : 24
Tensordot/ReadVariableOpTensordot/ReadVariableOp28
Tensordot_1/ReadVariableOpTensordot_1/ReadVariableOp28
Tensordot_2/ReadVariableOpTensordot_2/ReadVariableOp:P L

_output_shapes
:	dŹ
)
_user_specified_nameinputs_for_keys:RN

_output_shapes
:	dŹ
+
_user_specified_nameinputs_for_values:SO

_output_shapes
:	dŹ
,
_user_specified_nameinputs_for_queries
§

N__inference_transformer_block_layer_call_and_return_conditional_losses_3834247

inputs*
attention_head_3834132:
ŹŹ*
attention_head_3834134:
ŹŹ*
attention_head_3834136:
ŹŹ?
0layer_normalization_cast_readvariableop_resource:	ŹA
2layer_normalization_cast_1_readvariableop_resource:	ŹF
2sequential_dense_tensordot_readvariableop_resource:
ŹŹ?
0sequential_dense_biasadd_readvariableop_resource:	ŹH
4sequential_dense_1_tensordot_readvariableop_resource:
ŹŹA
2sequential_dense_1_biasadd_readvariableop_resource:	Ź
identity˘&attention_head/StatefulPartitionedCall˘'layer_normalization/Cast/ReadVariableOp˘)layer_normalization/Cast_1/ReadVariableOp˘)layer_normalization/Cast_2/ReadVariableOp˘)layer_normalization/Cast_3/ReadVariableOp˘'sequential/dense/BiasAdd/ReadVariableOp˘)sequential/dense/Tensordot/ReadVariableOp˘)sequential/dense_1/BiasAdd/ReadVariableOp˘+sequential/dense_1/Tensordot/ReadVariableOp
&attention_head/StatefulPartitionedCallStatefulPartitionedCallinputsinputsinputsattention_head_3834132attention_head_3834134attention_head_3834136*
Tin

2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:ddŹ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *!
fR
__inference_call_3827399s
addAddV2/attention_head/StatefulPartitionedCall:output:0inputs*
T0*#
_output_shapes
:ddŹn
layer_normalization/ShapeConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ,  q
'layer_normalization/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)layer_normalization/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)layer_normalization/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ľ
!layer_normalization/strided_sliceStridedSlice"layer_normalization/Shape:output:00layer_normalization/strided_slice/stack:output:02layer_normalization/strided_slice/stack_1:output:02layer_normalization/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask[
layer_normalization/mul/xConst*
_output_shapes
: *
dtype0*
value	B :
layer_normalization/mulMul"layer_normalization/mul/x:output:0*layer_normalization/strided_slice:output:0*
T0*
_output_shapes
: s
)layer_normalization/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:u
+layer_normalization/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+layer_normalization/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:˝
#layer_normalization/strided_slice_1StridedSlice"layer_normalization/Shape:output:02layer_normalization/strided_slice_1/stack:output:04layer_normalization/strided_slice_1/stack_1:output:04layer_normalization/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
layer_normalization/mul_1Mullayer_normalization/mul:z:0,layer_normalization/strided_slice_1:output:0*
T0*
_output_shapes
: s
)layer_normalization/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:u
+layer_normalization/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+layer_normalization/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:˝
#layer_normalization/strided_slice_2StridedSlice"layer_normalization/Shape:output:02layer_normalization/strided_slice_2/stack:output:04layer_normalization/strided_slice_2/stack_1:output:04layer_normalization/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
layer_normalization/mul_2/xConst*
_output_shapes
: *
dtype0*
value	B :
layer_normalization/mul_2Mul$layer_normalization/mul_2/x:output:0,layer_normalization/strided_slice_2:output:0*
T0*
_output_shapes
: e
#layer_normalization/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :e
#layer_normalization/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :ń
!layer_normalization/Reshape/shapePack,layer_normalization/Reshape/shape/0:output:0layer_normalization/mul_1:z:0layer_normalization/mul_2:z:0,layer_normalization/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:
layer_normalization/ReshapeReshapeadd:z:0*layer_normalization/Reshape/shape:output:0*
T0*(
_output_shapes
:NŹt
layer_normalization/ones/packedPacklayer_normalization/mul_1:z:0*
N*
T0*
_output_shapes
:c
layer_normalization/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
layer_normalization/onesFill(layer_normalization/ones/packed:output:0'layer_normalization/ones/Const:output:0*
T0*
_output_shapes	
:Nu
 layer_normalization/zeros/packedPacklayer_normalization/mul_1:z:0*
N*
T0*
_output_shapes
:d
layer_normalization/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
layer_normalization/zerosFill)layer_normalization/zeros/packed:output:0(layer_normalization/zeros/Const:output:0*
T0*
_output_shapes	
:N\
layer_normalization/ConstConst*
_output_shapes
: *
dtype0*
valueB ^
layer_normalization/Const_1Const*
_output_shapes
: *
dtype0*
valueB ô
$layer_normalization/FusedBatchNormV3FusedBatchNormV3$layer_normalization/Reshape:output:0!layer_normalization/ones:output:0"layer_normalization/zeros:output:0"layer_normalization/Const:output:0$layer_normalization/Const_1:output:0*
T0*
U0*H
_output_shapes6
4:NŹ:N:N:N:N:*
data_formatNCHW*
epsilon%o:¤
layer_normalization/Reshape_1Reshape(layer_normalization/FusedBatchNormV3:y:0"layer_normalization/Shape:output:0*
T0*#
_output_shapes
:ddŹ
'layer_normalization/Cast/ReadVariableOpReadVariableOp0layer_normalization_cast_readvariableop_resource*
_output_shapes	
:Ź*
dtype0§
layer_normalization/mul_3Mul&layer_normalization/Reshape_1:output:0/layer_normalization/Cast/ReadVariableOp:value:0*
T0*#
_output_shapes
:ddŹ
)layer_normalization/Cast_1/ReadVariableOpReadVariableOp2layer_normalization_cast_1_readvariableop_resource*
_output_shapes	
:Ź*
dtype0 
layer_normalization/addAddV2layer_normalization/mul_3:z:01layer_normalization/Cast_1/ReadVariableOp:value:0*
T0*#
_output_shapes
:ddŹ
)sequential/dense/Tensordot/ReadVariableOpReadVariableOp2sequential_dense_tensordot_readvariableop_resource* 
_output_shapes
:
ŹŹ*
dtype0y
(sequential/dense/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"'  ,  ¨
"sequential/dense/Tensordot/ReshapeReshapelayer_normalization/add:z:01sequential/dense/Tensordot/Reshape/shape:output:0*
T0* 
_output_shapes
:
NŹś
!sequential/dense/Tensordot/MatMulMatMul+sequential/dense/Tensordot/Reshape:output:01sequential/dense/Tensordot/ReadVariableOp:value:0*
T0* 
_output_shapes
:
NŹu
 sequential/dense/Tensordot/shapeConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ,  Ť
sequential/dense/TensordotReshape+sequential/dense/Tensordot/MatMul:product:0)sequential/dense/Tensordot/shape:output:0*
T0*#
_output_shapes
:ddŹ
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes	
:Ź*
dtype0§
sequential/dense/BiasAddBiasAdd#sequential/dense/Tensordot:output:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*#
_output_shapes
:ddŹo
sequential/dense/LeakyRelu	LeakyRelu!sequential/dense/BiasAdd:output:0*#
_output_shapes
:ddŹ˘
+sequential/dense_1/Tensordot/ReadVariableOpReadVariableOp4sequential_dense_1_tensordot_readvariableop_resource* 
_output_shapes
:
ŹŹ*
dtype0{
*sequential/dense_1/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"'  ,  š
$sequential/dense_1/Tensordot/ReshapeReshape(sequential/dense/LeakyRelu:activations:03sequential/dense_1/Tensordot/Reshape/shape:output:0*
T0* 
_output_shapes
:
NŹź
#sequential/dense_1/Tensordot/MatMulMatMul-sequential/dense_1/Tensordot/Reshape:output:03sequential/dense_1/Tensordot/ReadVariableOp:value:0*
T0* 
_output_shapes
:
NŹw
"sequential/dense_1/Tensordot/shapeConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ,  ą
sequential/dense_1/TensordotReshape-sequential/dense_1/Tensordot/MatMul:product:0+sequential/dense_1/Tensordot/shape:output:0*
T0*#
_output_shapes
:ddŹ
)sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:Ź*
dtype0­
sequential/dense_1/BiasAddBiasAdd%sequential/dense_1/Tensordot:output:01sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*#
_output_shapes
:ddŹ~
add_1AddV2#sequential/dense_1/BiasAdd:output:0layer_normalization/add:z:0*
T0*#
_output_shapes
:ddŹp
layer_normalization/Shape_1Const*
_output_shapes
:*
dtype0*!
valueB"d   d   ,  s
)layer_normalization/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+layer_normalization/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+layer_normalization/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ż
#layer_normalization/strided_slice_3StridedSlice$layer_normalization/Shape_1:output:02layer_normalization/strided_slice_3/stack:output:04layer_normalization/strided_slice_3/stack_1:output:04layer_normalization/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
layer_normalization/mul_4/xConst*
_output_shapes
: *
dtype0*
value	B :
layer_normalization/mul_4Mul$layer_normalization/mul_4/x:output:0,layer_normalization/strided_slice_3:output:0*
T0*
_output_shapes
: s
)layer_normalization/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:u
+layer_normalization/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+layer_normalization/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ż
#layer_normalization/strided_slice_4StridedSlice$layer_normalization/Shape_1:output:02layer_normalization/strided_slice_4/stack:output:04layer_normalization/strided_slice_4/stack_1:output:04layer_normalization/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
layer_normalization/mul_5Mullayer_normalization/mul_4:z:0,layer_normalization/strided_slice_4:output:0*
T0*
_output_shapes
: s
)layer_normalization/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:u
+layer_normalization/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+layer_normalization/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ż
#layer_normalization/strided_slice_5StridedSlice$layer_normalization/Shape_1:output:02layer_normalization/strided_slice_5/stack:output:04layer_normalization/strided_slice_5/stack_1:output:04layer_normalization/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
layer_normalization/mul_6/xConst*
_output_shapes
: *
dtype0*
value	B :
layer_normalization/mul_6Mul$layer_normalization/mul_6/x:output:0,layer_normalization/strided_slice_5:output:0*
T0*
_output_shapes
: g
%layer_normalization/Reshape_2/shape/0Const*
_output_shapes
: *
dtype0*
value	B :g
%layer_normalization/Reshape_2/shape/3Const*
_output_shapes
: *
dtype0*
value	B :÷
#layer_normalization/Reshape_2/shapePack.layer_normalization/Reshape_2/shape/0:output:0layer_normalization/mul_5:z:0layer_normalization/mul_6:z:0.layer_normalization/Reshape_2/shape/3:output:0*
N*
T0*
_output_shapes
:
layer_normalization/Reshape_2Reshape	add_1:z:0,layer_normalization/Reshape_2/shape:output:0*
T0*(
_output_shapes
:NŹv
!layer_normalization/ones_1/packedPacklayer_normalization/mul_5:z:0*
N*
T0*
_output_shapes
:e
 layer_normalization/ones_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
layer_normalization/ones_1Fill*layer_normalization/ones_1/packed:output:0)layer_normalization/ones_1/Const:output:0*
T0*
_output_shapes	
:Nw
"layer_normalization/zeros_1/packedPacklayer_normalization/mul_5:z:0*
N*
T0*
_output_shapes
:f
!layer_normalization/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ˘
layer_normalization/zeros_1Fill+layer_normalization/zeros_1/packed:output:0*layer_normalization/zeros_1/Const:output:0*
T0*
_output_shapes	
:N^
layer_normalization/Const_2Const*
_output_shapes
: *
dtype0*
valueB ^
layer_normalization/Const_3Const*
_output_shapes
: *
dtype0*
valueB ţ
&layer_normalization/FusedBatchNormV3_1FusedBatchNormV3&layer_normalization/Reshape_2:output:0#layer_normalization/ones_1:output:0$layer_normalization/zeros_1:output:0$layer_normalization/Const_2:output:0$layer_normalization/Const_3:output:0*
T0*
U0*H
_output_shapes6
4:NŹ:N:N:N:N:*
data_formatNCHW*
epsilon%o:¨
layer_normalization/Reshape_3Reshape*layer_normalization/FusedBatchNormV3_1:y:0$layer_normalization/Shape_1:output:0*
T0*#
_output_shapes
:ddŹ
)layer_normalization/Cast_2/ReadVariableOpReadVariableOp0layer_normalization_cast_readvariableop_resource*
_output_shapes	
:Ź*
dtype0Š
layer_normalization/mul_7Mul&layer_normalization/Reshape_3:output:01layer_normalization/Cast_2/ReadVariableOp:value:0*
T0*#
_output_shapes
:ddŹ
)layer_normalization/Cast_3/ReadVariableOpReadVariableOp2layer_normalization_cast_1_readvariableop_resource*
_output_shapes	
:Ź*
dtype0˘
layer_normalization/add_1AddV2layer_normalization/mul_7:z:01layer_normalization/Cast_3/ReadVariableOp:value:0*
T0*#
_output_shapes
:ddŹh
IdentityIdentitylayer_normalization/add_1:z:0^NoOp*
T0*#
_output_shapes
:ddŹÍ
NoOpNoOp'^attention_head/StatefulPartitionedCall(^layer_normalization/Cast/ReadVariableOp*^layer_normalization/Cast_1/ReadVariableOp*^layer_normalization/Cast_2/ReadVariableOp*^layer_normalization/Cast_3/ReadVariableOp(^sequential/dense/BiasAdd/ReadVariableOp*^sequential/dense/Tensordot/ReadVariableOp*^sequential/dense_1/BiasAdd/ReadVariableOp,^sequential/dense_1/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:	dŹ: : : : : : : : : 2P
&attention_head/StatefulPartitionedCall&attention_head/StatefulPartitionedCall2R
'layer_normalization/Cast/ReadVariableOp'layer_normalization/Cast/ReadVariableOp2V
)layer_normalization/Cast_1/ReadVariableOp)layer_normalization/Cast_1/ReadVariableOp2V
)layer_normalization/Cast_2/ReadVariableOp)layer_normalization/Cast_2/ReadVariableOp2V
)layer_normalization/Cast_3/ReadVariableOp)layer_normalization/Cast_3/ReadVariableOp2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2V
)sequential/dense/Tensordot/ReadVariableOp)sequential/dense/Tensordot/ReadVariableOp2V
)sequential/dense_1/BiasAdd/ReadVariableOp)sequential/dense_1/BiasAdd/ReadVariableOp2Z
+sequential/dense_1/Tensordot/ReadVariableOp+sequential/dense_1/Tensordot/ReadVariableOp:G C

_output_shapes
:	dŹ
 
_user_specified_nameinputs
łŔ
ç

P__inference_transformer_block_1_layer_call_and_return_conditional_losses_3834761

inputs
context_sequence,
attention_head_2_3834597:
ŹŹ,
attention_head_2_3834599:
ŹŹ,
attention_head_2_3834601:
ŹŹA
2layer_normalization_1_cast_readvariableop_resource:	ŹC
4layer_normalization_1_cast_1_readvariableop_resource:	Ź,
attention_head_3_3834648:
ŹŹ,
attention_head_3_3834650:
ŹŹ,
attention_head_3_3834652:
ŹŹJ
6sequential_1_dense_2_tensordot_readvariableop_resource:
ŹŹC
4sequential_1_dense_2_biasadd_readvariableop_resource:	ŹJ
6sequential_1_dense_3_tensordot_readvariableop_resource:
ŹŹC
4sequential_1_dense_3_biasadd_readvariableop_resource:	Ź
identity˘(attention_head_2/StatefulPartitionedCall˘(attention_head_3/StatefulPartitionedCall˘)layer_normalization_1/Cast/ReadVariableOp˘+layer_normalization_1/Cast_1/ReadVariableOp˘+layer_normalization_1/Cast_2/ReadVariableOp˘+layer_normalization_1/Cast_3/ReadVariableOp˘+layer_normalization_1/Cast_4/ReadVariableOp˘+layer_normalization_1/Cast_5/ReadVariableOp˘+sequential_1/dense_2/BiasAdd/ReadVariableOp˘-sequential_1/dense_2/Tensordot/ReadVariableOp˘+sequential_1/dense_3/BiasAdd/ReadVariableOp˘-sequential_1/dense_3/Tensordot/ReadVariableOp
(attention_head_2/StatefulPartitionedCallStatefulPartitionedCallinputsinputsinputsattention_head_2_3834597attention_head_2_3834599attention_head_2_3834601*
Tin

2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:ddŹ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *!
fR
__inference_call_3827587u
addAddV21attention_head_2/StatefulPartitionedCall:output:0inputs*
T0*#
_output_shapes
:ddŹp
layer_normalization_1/ShapeConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ,  s
)layer_normalization_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+layer_normalization_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+layer_normalization_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ż
#layer_normalization_1/strided_sliceStridedSlice$layer_normalization_1/Shape:output:02layer_normalization_1/strided_slice/stack:output:04layer_normalization_1/strided_slice/stack_1:output:04layer_normalization_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
layer_normalization_1/mul/xConst*
_output_shapes
: *
dtype0*
value	B :
layer_normalization_1/mulMul$layer_normalization_1/mul/x:output:0,layer_normalization_1/strided_slice:output:0*
T0*
_output_shapes
: u
+layer_normalization_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ç
%layer_normalization_1/strided_slice_1StridedSlice$layer_normalization_1/Shape:output:04layer_normalization_1/strided_slice_1/stack:output:06layer_normalization_1/strided_slice_1/stack_1:output:06layer_normalization_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
layer_normalization_1/mul_1Mullayer_normalization_1/mul:z:0.layer_normalization_1/strided_slice_1:output:0*
T0*
_output_shapes
: u
+layer_normalization_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ç
%layer_normalization_1/strided_slice_2StridedSlice$layer_normalization_1/Shape:output:04layer_normalization_1/strided_slice_2/stack:output:06layer_normalization_1/strided_slice_2/stack_1:output:06layer_normalization_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
layer_normalization_1/mul_2/xConst*
_output_shapes
: *
dtype0*
value	B :
layer_normalization_1/mul_2Mul&layer_normalization_1/mul_2/x:output:0.layer_normalization_1/strided_slice_2:output:0*
T0*
_output_shapes
: g
%layer_normalization_1/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :g
%layer_normalization_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :ű
#layer_normalization_1/Reshape/shapePack.layer_normalization_1/Reshape/shape/0:output:0layer_normalization_1/mul_1:z:0layer_normalization_1/mul_2:z:0.layer_normalization_1/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:
layer_normalization_1/ReshapeReshapeadd:z:0,layer_normalization_1/Reshape/shape:output:0*
T0*(
_output_shapes
:NŹx
!layer_normalization_1/ones/packedPacklayer_normalization_1/mul_1:z:0*
N*
T0*
_output_shapes
:e
 layer_normalization_1/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
layer_normalization_1/onesFill*layer_normalization_1/ones/packed:output:0)layer_normalization_1/ones/Const:output:0*
T0*
_output_shapes	
:Ny
"layer_normalization_1/zeros/packedPacklayer_normalization_1/mul_1:z:0*
N*
T0*
_output_shapes
:f
!layer_normalization_1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ˘
layer_normalization_1/zerosFill+layer_normalization_1/zeros/packed:output:0*layer_normalization_1/zeros/Const:output:0*
T0*
_output_shapes	
:N^
layer_normalization_1/ConstConst*
_output_shapes
: *
dtype0*
valueB `
layer_normalization_1/Const_1Const*
_output_shapes
: *
dtype0*
valueB 
&layer_normalization_1/FusedBatchNormV3FusedBatchNormV3&layer_normalization_1/Reshape:output:0#layer_normalization_1/ones:output:0$layer_normalization_1/zeros:output:0$layer_normalization_1/Const:output:0&layer_normalization_1/Const_1:output:0*
T0*
U0*H
_output_shapes6
4:NŹ:N:N:N:N:*
data_formatNCHW*
epsilon%o:Ş
layer_normalization_1/Reshape_1Reshape*layer_normalization_1/FusedBatchNormV3:y:0$layer_normalization_1/Shape:output:0*
T0*#
_output_shapes
:ddŹ
)layer_normalization_1/Cast/ReadVariableOpReadVariableOp2layer_normalization_1_cast_readvariableop_resource*
_output_shapes	
:Ź*
dtype0­
layer_normalization_1/mul_3Mul(layer_normalization_1/Reshape_1:output:01layer_normalization_1/Cast/ReadVariableOp:value:0*
T0*#
_output_shapes
:ddŹ
+layer_normalization_1/Cast_1/ReadVariableOpReadVariableOp4layer_normalization_1_cast_1_readvariableop_resource*
_output_shapes	
:Ź*
dtype0Ś
layer_normalization_1/addAddV2layer_normalization_1/mul_3:z:03layer_normalization_1/Cast_1/ReadVariableOp:value:0*
T0*#
_output_shapes
:ddŹŔ
(attention_head_3/StatefulPartitionedCallStatefulPartitionedCalllayer_normalization_1/add:z:0layer_normalization_1/add:z:0context_sequenceattention_head_3_3834648attention_head_3_3834650attention_head_3_3834652*
Tin

2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:ddŹ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *!
fR
__inference_call_3827682
add_1AddV21attention_head_3/StatefulPartitionedCall:output:0layer_normalization_1/add:z:0*
T0*#
_output_shapes
:ddŹr
layer_normalization_1/Shape_1Const*
_output_shapes
:*
dtype0*!
valueB"d   d   ,  u
+layer_normalization_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-layer_normalization_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:É
%layer_normalization_1/strided_slice_3StridedSlice&layer_normalization_1/Shape_1:output:04layer_normalization_1/strided_slice_3/stack:output:06layer_normalization_1/strided_slice_3/stack_1:output:06layer_normalization_1/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
layer_normalization_1/mul_4/xConst*
_output_shapes
: *
dtype0*
value	B :
layer_normalization_1/mul_4Mul&layer_normalization_1/mul_4/x:output:0.layer_normalization_1/strided_slice_3:output:0*
T0*
_output_shapes
: u
+layer_normalization_1/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_1/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_1/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:É
%layer_normalization_1/strided_slice_4StridedSlice&layer_normalization_1/Shape_1:output:04layer_normalization_1/strided_slice_4/stack:output:06layer_normalization_1/strided_slice_4/stack_1:output:06layer_normalization_1/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
layer_normalization_1/mul_5Mullayer_normalization_1/mul_4:z:0.layer_normalization_1/strided_slice_4:output:0*
T0*
_output_shapes
: u
+layer_normalization_1/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_1/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_1/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:É
%layer_normalization_1/strided_slice_5StridedSlice&layer_normalization_1/Shape_1:output:04layer_normalization_1/strided_slice_5/stack:output:06layer_normalization_1/strided_slice_5/stack_1:output:06layer_normalization_1/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
layer_normalization_1/mul_6/xConst*
_output_shapes
: *
dtype0*
value	B :
layer_normalization_1/mul_6Mul&layer_normalization_1/mul_6/x:output:0.layer_normalization_1/strided_slice_5:output:0*
T0*
_output_shapes
: i
'layer_normalization_1/Reshape_2/shape/0Const*
_output_shapes
: *
dtype0*
value	B :i
'layer_normalization_1/Reshape_2/shape/3Const*
_output_shapes
: *
dtype0*
value	B :
%layer_normalization_1/Reshape_2/shapePack0layer_normalization_1/Reshape_2/shape/0:output:0layer_normalization_1/mul_5:z:0layer_normalization_1/mul_6:z:00layer_normalization_1/Reshape_2/shape/3:output:0*
N*
T0*
_output_shapes
:
layer_normalization_1/Reshape_2Reshape	add_1:z:0.layer_normalization_1/Reshape_2/shape:output:0*
T0*(
_output_shapes
:NŹz
#layer_normalization_1/ones_1/packedPacklayer_normalization_1/mul_5:z:0*
N*
T0*
_output_shapes
:g
"layer_normalization_1/ones_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ľ
layer_normalization_1/ones_1Fill,layer_normalization_1/ones_1/packed:output:0+layer_normalization_1/ones_1/Const:output:0*
T0*
_output_shapes	
:N{
$layer_normalization_1/zeros_1/packedPacklayer_normalization_1/mul_5:z:0*
N*
T0*
_output_shapes
:h
#layer_normalization_1/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ¨
layer_normalization_1/zeros_1Fill-layer_normalization_1/zeros_1/packed:output:0,layer_normalization_1/zeros_1/Const:output:0*
T0*
_output_shapes	
:N`
layer_normalization_1/Const_2Const*
_output_shapes
: *
dtype0*
valueB `
layer_normalization_1/Const_3Const*
_output_shapes
: *
dtype0*
valueB 
(layer_normalization_1/FusedBatchNormV3_1FusedBatchNormV3(layer_normalization_1/Reshape_2:output:0%layer_normalization_1/ones_1:output:0&layer_normalization_1/zeros_1:output:0&layer_normalization_1/Const_2:output:0&layer_normalization_1/Const_3:output:0*
T0*
U0*H
_output_shapes6
4:NŹ:N:N:N:N:*
data_formatNCHW*
epsilon%o:Ž
layer_normalization_1/Reshape_3Reshape,layer_normalization_1/FusedBatchNormV3_1:y:0&layer_normalization_1/Shape_1:output:0*
T0*#
_output_shapes
:ddŹ
+layer_normalization_1/Cast_2/ReadVariableOpReadVariableOp2layer_normalization_1_cast_readvariableop_resource*
_output_shapes	
:Ź*
dtype0Ż
layer_normalization_1/mul_7Mul(layer_normalization_1/Reshape_3:output:03layer_normalization_1/Cast_2/ReadVariableOp:value:0*
T0*#
_output_shapes
:ddŹ
+layer_normalization_1/Cast_3/ReadVariableOpReadVariableOp4layer_normalization_1_cast_1_readvariableop_resource*
_output_shapes	
:Ź*
dtype0¨
layer_normalization_1/add_1AddV2layer_normalization_1/mul_7:z:03layer_normalization_1/Cast_3/ReadVariableOp:value:0*
T0*#
_output_shapes
:ddŹŚ
-sequential_1/dense_2/Tensordot/ReadVariableOpReadVariableOp6sequential_1_dense_2_tensordot_readvariableop_resource* 
_output_shapes
:
ŹŹ*
dtype0}
,sequential_1/dense_2/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"'  ,  ´
&sequential_1/dense_2/Tensordot/ReshapeReshapelayer_normalization_1/add_1:z:05sequential_1/dense_2/Tensordot/Reshape/shape:output:0*
T0* 
_output_shapes
:
NŹÂ
%sequential_1/dense_2/Tensordot/MatMulMatMul/sequential_1/dense_2/Tensordot/Reshape:output:05sequential_1/dense_2/Tensordot/ReadVariableOp:value:0*
T0* 
_output_shapes
:
NŹy
$sequential_1/dense_2/Tensordot/shapeConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ,  ˇ
sequential_1/dense_2/TensordotReshape/sequential_1/dense_2/Tensordot/MatMul:product:0-sequential_1/dense_2/Tensordot/shape:output:0*
T0*#
_output_shapes
:ddŹ
+sequential_1/dense_2/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:Ź*
dtype0ł
sequential_1/dense_2/BiasAddBiasAdd'sequential_1/dense_2/Tensordot:output:03sequential_1/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*#
_output_shapes
:ddŹw
sequential_1/dense_2/LeakyRelu	LeakyRelu%sequential_1/dense_2/BiasAdd:output:0*#
_output_shapes
:ddŹŚ
-sequential_1/dense_3/Tensordot/ReadVariableOpReadVariableOp6sequential_1_dense_3_tensordot_readvariableop_resource* 
_output_shapes
:
ŹŹ*
dtype0}
,sequential_1/dense_3/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"'  ,  Á
&sequential_1/dense_3/Tensordot/ReshapeReshape,sequential_1/dense_2/LeakyRelu:activations:05sequential_1/dense_3/Tensordot/Reshape/shape:output:0*
T0* 
_output_shapes
:
NŹÂ
%sequential_1/dense_3/Tensordot/MatMulMatMul/sequential_1/dense_3/Tensordot/Reshape:output:05sequential_1/dense_3/Tensordot/ReadVariableOp:value:0*
T0* 
_output_shapes
:
NŹy
$sequential_1/dense_3/Tensordot/shapeConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ,  ˇ
sequential_1/dense_3/TensordotReshape/sequential_1/dense_3/Tensordot/MatMul:product:0-sequential_1/dense_3/Tensordot/shape:output:0*
T0*#
_output_shapes
:ddŹ
+sequential_1/dense_3/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_3_biasadd_readvariableop_resource*
_output_shapes	
:Ź*
dtype0ł
sequential_1/dense_3/BiasAddBiasAdd'sequential_1/dense_3/Tensordot:output:03sequential_1/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*#
_output_shapes
:ddŹ
add_2AddV2%sequential_1/dense_3/BiasAdd:output:0layer_normalization_1/add_1:z:0*
T0*#
_output_shapes
:ddŹr
layer_normalization_1/Shape_2Const*
_output_shapes
:*
dtype0*!
valueB"d   d   ,  u
+layer_normalization_1/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-layer_normalization_1/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_1/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB:É
%layer_normalization_1/strided_slice_6StridedSlice&layer_normalization_1/Shape_2:output:04layer_normalization_1/strided_slice_6/stack:output:06layer_normalization_1/strided_slice_6/stack_1:output:06layer_normalization_1/strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
layer_normalization_1/mul_8/xConst*
_output_shapes
: *
dtype0*
value	B :
layer_normalization_1/mul_8Mul&layer_normalization_1/mul_8/x:output:0.layer_normalization_1/strided_slice_6:output:0*
T0*
_output_shapes
: u
+layer_normalization_1/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_1/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_1/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB:É
%layer_normalization_1/strided_slice_7StridedSlice&layer_normalization_1/Shape_2:output:04layer_normalization_1/strided_slice_7/stack:output:06layer_normalization_1/strided_slice_7/stack_1:output:06layer_normalization_1/strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
layer_normalization_1/mul_9Mullayer_normalization_1/mul_8:z:0.layer_normalization_1/strided_slice_7:output:0*
T0*
_output_shapes
: u
+layer_normalization_1/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_1/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_1/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:É
%layer_normalization_1/strided_slice_8StridedSlice&layer_normalization_1/Shape_2:output:04layer_normalization_1/strided_slice_8/stack:output:06layer_normalization_1/strided_slice_8/stack_1:output:06layer_normalization_1/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask`
layer_normalization_1/mul_10/xConst*
_output_shapes
: *
dtype0*
value	B :
layer_normalization_1/mul_10Mul'layer_normalization_1/mul_10/x:output:0.layer_normalization_1/strided_slice_8:output:0*
T0*
_output_shapes
: i
'layer_normalization_1/Reshape_4/shape/0Const*
_output_shapes
: *
dtype0*
value	B :i
'layer_normalization_1/Reshape_4/shape/3Const*
_output_shapes
: *
dtype0*
value	B :
%layer_normalization_1/Reshape_4/shapePack0layer_normalization_1/Reshape_4/shape/0:output:0layer_normalization_1/mul_9:z:0 layer_normalization_1/mul_10:z:00layer_normalization_1/Reshape_4/shape/3:output:0*
N*
T0*
_output_shapes
:
layer_normalization_1/Reshape_4Reshape	add_2:z:0.layer_normalization_1/Reshape_4/shape:output:0*
T0*(
_output_shapes
:NŹz
#layer_normalization_1/ones_2/packedPacklayer_normalization_1/mul_9:z:0*
N*
T0*
_output_shapes
:g
"layer_normalization_1/ones_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ľ
layer_normalization_1/ones_2Fill,layer_normalization_1/ones_2/packed:output:0+layer_normalization_1/ones_2/Const:output:0*
T0*
_output_shapes	
:N{
$layer_normalization_1/zeros_2/packedPacklayer_normalization_1/mul_9:z:0*
N*
T0*
_output_shapes
:h
#layer_normalization_1/zeros_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ¨
layer_normalization_1/zeros_2Fill-layer_normalization_1/zeros_2/packed:output:0,layer_normalization_1/zeros_2/Const:output:0*
T0*
_output_shapes	
:N`
layer_normalization_1/Const_4Const*
_output_shapes
: *
dtype0*
valueB `
layer_normalization_1/Const_5Const*
_output_shapes
: *
dtype0*
valueB 
(layer_normalization_1/FusedBatchNormV3_2FusedBatchNormV3(layer_normalization_1/Reshape_4:output:0%layer_normalization_1/ones_2:output:0&layer_normalization_1/zeros_2:output:0&layer_normalization_1/Const_4:output:0&layer_normalization_1/Const_5:output:0*
T0*
U0*H
_output_shapes6
4:NŹ:N:N:N:N:*
data_formatNCHW*
epsilon%o:Ž
layer_normalization_1/Reshape_5Reshape,layer_normalization_1/FusedBatchNormV3_2:y:0&layer_normalization_1/Shape_2:output:0*
T0*#
_output_shapes
:ddŹ
+layer_normalization_1/Cast_4/ReadVariableOpReadVariableOp2layer_normalization_1_cast_readvariableop_resource*
_output_shapes	
:Ź*
dtype0°
layer_normalization_1/mul_11Mul(layer_normalization_1/Reshape_5:output:03layer_normalization_1/Cast_4/ReadVariableOp:value:0*
T0*#
_output_shapes
:ddŹ
+layer_normalization_1/Cast_5/ReadVariableOpReadVariableOp4layer_normalization_1_cast_1_readvariableop_resource*
_output_shapes	
:Ź*
dtype0Š
layer_normalization_1/add_2AddV2 layer_normalization_1/mul_11:z:03layer_normalization_1/Cast_5/ReadVariableOp:value:0*
T0*#
_output_shapes
:ddŹj
IdentityIdentitylayer_normalization_1/add_2:z:0^NoOp*
T0*#
_output_shapes
:ddŹę
NoOpNoOp)^attention_head_2/StatefulPartitionedCall)^attention_head_3/StatefulPartitionedCall*^layer_normalization_1/Cast/ReadVariableOp,^layer_normalization_1/Cast_1/ReadVariableOp,^layer_normalization_1/Cast_2/ReadVariableOp,^layer_normalization_1/Cast_3/ReadVariableOp,^layer_normalization_1/Cast_4/ReadVariableOp,^layer_normalization_1/Cast_5/ReadVariableOp,^sequential_1/dense_2/BiasAdd/ReadVariableOp.^sequential_1/dense_2/Tensordot/ReadVariableOp,^sequential_1/dense_3/BiasAdd/ReadVariableOp.^sequential_1/dense_3/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:ddŹ:	dŹ: : : : : : : : : : : : 2T
(attention_head_2/StatefulPartitionedCall(attention_head_2/StatefulPartitionedCall2T
(attention_head_3/StatefulPartitionedCall(attention_head_3/StatefulPartitionedCall2V
)layer_normalization_1/Cast/ReadVariableOp)layer_normalization_1/Cast/ReadVariableOp2Z
+layer_normalization_1/Cast_1/ReadVariableOp+layer_normalization_1/Cast_1/ReadVariableOp2Z
+layer_normalization_1/Cast_2/ReadVariableOp+layer_normalization_1/Cast_2/ReadVariableOp2Z
+layer_normalization_1/Cast_3/ReadVariableOp+layer_normalization_1/Cast_3/ReadVariableOp2Z
+layer_normalization_1/Cast_4/ReadVariableOp+layer_normalization_1/Cast_4/ReadVariableOp2Z
+layer_normalization_1/Cast_5/ReadVariableOp+layer_normalization_1/Cast_5/ReadVariableOp2Z
+sequential_1/dense_2/BiasAdd/ReadVariableOp+sequential_1/dense_2/BiasAdd/ReadVariableOp2^
-sequential_1/dense_2/Tensordot/ReadVariableOp-sequential_1/dense_2/Tensordot/ReadVariableOp2Z
+sequential_1/dense_3/BiasAdd/ReadVariableOp+sequential_1/dense_3/BiasAdd/ReadVariableOp2^
-sequential_1/dense_3/Tensordot/ReadVariableOp-sequential_1/dense_3/Tensordot/ReadVariableOp:K G
#
_output_shapes
:ddŹ
 
_user_specified_nameinputs:QM

_output_shapes
:	dŹ
*
_user_specified_namecontext_sequence
Ů

)__inference_dense_1_layer_call_fn_3835188

inputs
unknown:
ŹŹ
	unknown_0:	Ź
identity˘StatefulPartitionedCallŢ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:˙˙˙˙˙˙˙˙˙dŹ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_3832067t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙dŹ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:˙˙˙˙˙˙˙˙˙dŹ: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:˙˙˙˙˙˙˙˙˙dŹ
 
_user_specified_nameinputs
łŔ
ç

P__inference_transformer_block_1_layer_call_and_return_conditional_losses_3834593

inputs
context_sequence,
attention_head_2_3834429:
ŹŹ,
attention_head_2_3834431:
ŹŹ,
attention_head_2_3834433:
ŹŹA
2layer_normalization_1_cast_readvariableop_resource:	ŹC
4layer_normalization_1_cast_1_readvariableop_resource:	Ź,
attention_head_3_3834480:
ŹŹ,
attention_head_3_3834482:
ŹŹ,
attention_head_3_3834484:
ŹŹJ
6sequential_1_dense_2_tensordot_readvariableop_resource:
ŹŹC
4sequential_1_dense_2_biasadd_readvariableop_resource:	ŹJ
6sequential_1_dense_3_tensordot_readvariableop_resource:
ŹŹC
4sequential_1_dense_3_biasadd_readvariableop_resource:	Ź
identity˘(attention_head_2/StatefulPartitionedCall˘(attention_head_3/StatefulPartitionedCall˘)layer_normalization_1/Cast/ReadVariableOp˘+layer_normalization_1/Cast_1/ReadVariableOp˘+layer_normalization_1/Cast_2/ReadVariableOp˘+layer_normalization_1/Cast_3/ReadVariableOp˘+layer_normalization_1/Cast_4/ReadVariableOp˘+layer_normalization_1/Cast_5/ReadVariableOp˘+sequential_1/dense_2/BiasAdd/ReadVariableOp˘-sequential_1/dense_2/Tensordot/ReadVariableOp˘+sequential_1/dense_3/BiasAdd/ReadVariableOp˘-sequential_1/dense_3/Tensordot/ReadVariableOp
(attention_head_2/StatefulPartitionedCallStatefulPartitionedCallinputsinputsinputsattention_head_2_3834429attention_head_2_3834431attention_head_2_3834433*
Tin

2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:ddŹ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *!
fR
__inference_call_3827587u
addAddV21attention_head_2/StatefulPartitionedCall:output:0inputs*
T0*#
_output_shapes
:ddŹp
layer_normalization_1/ShapeConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ,  s
)layer_normalization_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+layer_normalization_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+layer_normalization_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ż
#layer_normalization_1/strided_sliceStridedSlice$layer_normalization_1/Shape:output:02layer_normalization_1/strided_slice/stack:output:04layer_normalization_1/strided_slice/stack_1:output:04layer_normalization_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
layer_normalization_1/mul/xConst*
_output_shapes
: *
dtype0*
value	B :
layer_normalization_1/mulMul$layer_normalization_1/mul/x:output:0,layer_normalization_1/strided_slice:output:0*
T0*
_output_shapes
: u
+layer_normalization_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ç
%layer_normalization_1/strided_slice_1StridedSlice$layer_normalization_1/Shape:output:04layer_normalization_1/strided_slice_1/stack:output:06layer_normalization_1/strided_slice_1/stack_1:output:06layer_normalization_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
layer_normalization_1/mul_1Mullayer_normalization_1/mul:z:0.layer_normalization_1/strided_slice_1:output:0*
T0*
_output_shapes
: u
+layer_normalization_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ç
%layer_normalization_1/strided_slice_2StridedSlice$layer_normalization_1/Shape:output:04layer_normalization_1/strided_slice_2/stack:output:06layer_normalization_1/strided_slice_2/stack_1:output:06layer_normalization_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
layer_normalization_1/mul_2/xConst*
_output_shapes
: *
dtype0*
value	B :
layer_normalization_1/mul_2Mul&layer_normalization_1/mul_2/x:output:0.layer_normalization_1/strided_slice_2:output:0*
T0*
_output_shapes
: g
%layer_normalization_1/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :g
%layer_normalization_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :ű
#layer_normalization_1/Reshape/shapePack.layer_normalization_1/Reshape/shape/0:output:0layer_normalization_1/mul_1:z:0layer_normalization_1/mul_2:z:0.layer_normalization_1/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:
layer_normalization_1/ReshapeReshapeadd:z:0,layer_normalization_1/Reshape/shape:output:0*
T0*(
_output_shapes
:NŹx
!layer_normalization_1/ones/packedPacklayer_normalization_1/mul_1:z:0*
N*
T0*
_output_shapes
:e
 layer_normalization_1/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
layer_normalization_1/onesFill*layer_normalization_1/ones/packed:output:0)layer_normalization_1/ones/Const:output:0*
T0*
_output_shapes	
:Ny
"layer_normalization_1/zeros/packedPacklayer_normalization_1/mul_1:z:0*
N*
T0*
_output_shapes
:f
!layer_normalization_1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ˘
layer_normalization_1/zerosFill+layer_normalization_1/zeros/packed:output:0*layer_normalization_1/zeros/Const:output:0*
T0*
_output_shapes	
:N^
layer_normalization_1/ConstConst*
_output_shapes
: *
dtype0*
valueB `
layer_normalization_1/Const_1Const*
_output_shapes
: *
dtype0*
valueB 
&layer_normalization_1/FusedBatchNormV3FusedBatchNormV3&layer_normalization_1/Reshape:output:0#layer_normalization_1/ones:output:0$layer_normalization_1/zeros:output:0$layer_normalization_1/Const:output:0&layer_normalization_1/Const_1:output:0*
T0*
U0*H
_output_shapes6
4:NŹ:N:N:N:N:*
data_formatNCHW*
epsilon%o:Ş
layer_normalization_1/Reshape_1Reshape*layer_normalization_1/FusedBatchNormV3:y:0$layer_normalization_1/Shape:output:0*
T0*#
_output_shapes
:ddŹ
)layer_normalization_1/Cast/ReadVariableOpReadVariableOp2layer_normalization_1_cast_readvariableop_resource*
_output_shapes	
:Ź*
dtype0­
layer_normalization_1/mul_3Mul(layer_normalization_1/Reshape_1:output:01layer_normalization_1/Cast/ReadVariableOp:value:0*
T0*#
_output_shapes
:ddŹ
+layer_normalization_1/Cast_1/ReadVariableOpReadVariableOp4layer_normalization_1_cast_1_readvariableop_resource*
_output_shapes	
:Ź*
dtype0Ś
layer_normalization_1/addAddV2layer_normalization_1/mul_3:z:03layer_normalization_1/Cast_1/ReadVariableOp:value:0*
T0*#
_output_shapes
:ddŹŔ
(attention_head_3/StatefulPartitionedCallStatefulPartitionedCalllayer_normalization_1/add:z:0layer_normalization_1/add:z:0context_sequenceattention_head_3_3834480attention_head_3_3834482attention_head_3_3834484*
Tin

2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:ddŹ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *!
fR
__inference_call_3827682
add_1AddV21attention_head_3/StatefulPartitionedCall:output:0layer_normalization_1/add:z:0*
T0*#
_output_shapes
:ddŹr
layer_normalization_1/Shape_1Const*
_output_shapes
:*
dtype0*!
valueB"d   d   ,  u
+layer_normalization_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-layer_normalization_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:É
%layer_normalization_1/strided_slice_3StridedSlice&layer_normalization_1/Shape_1:output:04layer_normalization_1/strided_slice_3/stack:output:06layer_normalization_1/strided_slice_3/stack_1:output:06layer_normalization_1/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
layer_normalization_1/mul_4/xConst*
_output_shapes
: *
dtype0*
value	B :
layer_normalization_1/mul_4Mul&layer_normalization_1/mul_4/x:output:0.layer_normalization_1/strided_slice_3:output:0*
T0*
_output_shapes
: u
+layer_normalization_1/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_1/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_1/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:É
%layer_normalization_1/strided_slice_4StridedSlice&layer_normalization_1/Shape_1:output:04layer_normalization_1/strided_slice_4/stack:output:06layer_normalization_1/strided_slice_4/stack_1:output:06layer_normalization_1/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
layer_normalization_1/mul_5Mullayer_normalization_1/mul_4:z:0.layer_normalization_1/strided_slice_4:output:0*
T0*
_output_shapes
: u
+layer_normalization_1/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_1/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_1/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:É
%layer_normalization_1/strided_slice_5StridedSlice&layer_normalization_1/Shape_1:output:04layer_normalization_1/strided_slice_5/stack:output:06layer_normalization_1/strided_slice_5/stack_1:output:06layer_normalization_1/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
layer_normalization_1/mul_6/xConst*
_output_shapes
: *
dtype0*
value	B :
layer_normalization_1/mul_6Mul&layer_normalization_1/mul_6/x:output:0.layer_normalization_1/strided_slice_5:output:0*
T0*
_output_shapes
: i
'layer_normalization_1/Reshape_2/shape/0Const*
_output_shapes
: *
dtype0*
value	B :i
'layer_normalization_1/Reshape_2/shape/3Const*
_output_shapes
: *
dtype0*
value	B :
%layer_normalization_1/Reshape_2/shapePack0layer_normalization_1/Reshape_2/shape/0:output:0layer_normalization_1/mul_5:z:0layer_normalization_1/mul_6:z:00layer_normalization_1/Reshape_2/shape/3:output:0*
N*
T0*
_output_shapes
:
layer_normalization_1/Reshape_2Reshape	add_1:z:0.layer_normalization_1/Reshape_2/shape:output:0*
T0*(
_output_shapes
:NŹz
#layer_normalization_1/ones_1/packedPacklayer_normalization_1/mul_5:z:0*
N*
T0*
_output_shapes
:g
"layer_normalization_1/ones_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ľ
layer_normalization_1/ones_1Fill,layer_normalization_1/ones_1/packed:output:0+layer_normalization_1/ones_1/Const:output:0*
T0*
_output_shapes	
:N{
$layer_normalization_1/zeros_1/packedPacklayer_normalization_1/mul_5:z:0*
N*
T0*
_output_shapes
:h
#layer_normalization_1/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ¨
layer_normalization_1/zeros_1Fill-layer_normalization_1/zeros_1/packed:output:0,layer_normalization_1/zeros_1/Const:output:0*
T0*
_output_shapes	
:N`
layer_normalization_1/Const_2Const*
_output_shapes
: *
dtype0*
valueB `
layer_normalization_1/Const_3Const*
_output_shapes
: *
dtype0*
valueB 
(layer_normalization_1/FusedBatchNormV3_1FusedBatchNormV3(layer_normalization_1/Reshape_2:output:0%layer_normalization_1/ones_1:output:0&layer_normalization_1/zeros_1:output:0&layer_normalization_1/Const_2:output:0&layer_normalization_1/Const_3:output:0*
T0*
U0*H
_output_shapes6
4:NŹ:N:N:N:N:*
data_formatNCHW*
epsilon%o:Ž
layer_normalization_1/Reshape_3Reshape,layer_normalization_1/FusedBatchNormV3_1:y:0&layer_normalization_1/Shape_1:output:0*
T0*#
_output_shapes
:ddŹ
+layer_normalization_1/Cast_2/ReadVariableOpReadVariableOp2layer_normalization_1_cast_readvariableop_resource*
_output_shapes	
:Ź*
dtype0Ż
layer_normalization_1/mul_7Mul(layer_normalization_1/Reshape_3:output:03layer_normalization_1/Cast_2/ReadVariableOp:value:0*
T0*#
_output_shapes
:ddŹ
+layer_normalization_1/Cast_3/ReadVariableOpReadVariableOp4layer_normalization_1_cast_1_readvariableop_resource*
_output_shapes	
:Ź*
dtype0¨
layer_normalization_1/add_1AddV2layer_normalization_1/mul_7:z:03layer_normalization_1/Cast_3/ReadVariableOp:value:0*
T0*#
_output_shapes
:ddŹŚ
-sequential_1/dense_2/Tensordot/ReadVariableOpReadVariableOp6sequential_1_dense_2_tensordot_readvariableop_resource* 
_output_shapes
:
ŹŹ*
dtype0}
,sequential_1/dense_2/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"'  ,  ´
&sequential_1/dense_2/Tensordot/ReshapeReshapelayer_normalization_1/add_1:z:05sequential_1/dense_2/Tensordot/Reshape/shape:output:0*
T0* 
_output_shapes
:
NŹÂ
%sequential_1/dense_2/Tensordot/MatMulMatMul/sequential_1/dense_2/Tensordot/Reshape:output:05sequential_1/dense_2/Tensordot/ReadVariableOp:value:0*
T0* 
_output_shapes
:
NŹy
$sequential_1/dense_2/Tensordot/shapeConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ,  ˇ
sequential_1/dense_2/TensordotReshape/sequential_1/dense_2/Tensordot/MatMul:product:0-sequential_1/dense_2/Tensordot/shape:output:0*
T0*#
_output_shapes
:ddŹ
+sequential_1/dense_2/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:Ź*
dtype0ł
sequential_1/dense_2/BiasAddBiasAdd'sequential_1/dense_2/Tensordot:output:03sequential_1/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*#
_output_shapes
:ddŹw
sequential_1/dense_2/LeakyRelu	LeakyRelu%sequential_1/dense_2/BiasAdd:output:0*#
_output_shapes
:ddŹŚ
-sequential_1/dense_3/Tensordot/ReadVariableOpReadVariableOp6sequential_1_dense_3_tensordot_readvariableop_resource* 
_output_shapes
:
ŹŹ*
dtype0}
,sequential_1/dense_3/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"'  ,  Á
&sequential_1/dense_3/Tensordot/ReshapeReshape,sequential_1/dense_2/LeakyRelu:activations:05sequential_1/dense_3/Tensordot/Reshape/shape:output:0*
T0* 
_output_shapes
:
NŹÂ
%sequential_1/dense_3/Tensordot/MatMulMatMul/sequential_1/dense_3/Tensordot/Reshape:output:05sequential_1/dense_3/Tensordot/ReadVariableOp:value:0*
T0* 
_output_shapes
:
NŹy
$sequential_1/dense_3/Tensordot/shapeConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ,  ˇ
sequential_1/dense_3/TensordotReshape/sequential_1/dense_3/Tensordot/MatMul:product:0-sequential_1/dense_3/Tensordot/shape:output:0*
T0*#
_output_shapes
:ddŹ
+sequential_1/dense_3/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_3_biasadd_readvariableop_resource*
_output_shapes	
:Ź*
dtype0ł
sequential_1/dense_3/BiasAddBiasAdd'sequential_1/dense_3/Tensordot:output:03sequential_1/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*#
_output_shapes
:ddŹ
add_2AddV2%sequential_1/dense_3/BiasAdd:output:0layer_normalization_1/add_1:z:0*
T0*#
_output_shapes
:ddŹr
layer_normalization_1/Shape_2Const*
_output_shapes
:*
dtype0*!
valueB"d   d   ,  u
+layer_normalization_1/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-layer_normalization_1/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_1/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB:É
%layer_normalization_1/strided_slice_6StridedSlice&layer_normalization_1/Shape_2:output:04layer_normalization_1/strided_slice_6/stack:output:06layer_normalization_1/strided_slice_6/stack_1:output:06layer_normalization_1/strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
layer_normalization_1/mul_8/xConst*
_output_shapes
: *
dtype0*
value	B :
layer_normalization_1/mul_8Mul&layer_normalization_1/mul_8/x:output:0.layer_normalization_1/strided_slice_6:output:0*
T0*
_output_shapes
: u
+layer_normalization_1/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_1/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_1/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB:É
%layer_normalization_1/strided_slice_7StridedSlice&layer_normalization_1/Shape_2:output:04layer_normalization_1/strided_slice_7/stack:output:06layer_normalization_1/strided_slice_7/stack_1:output:06layer_normalization_1/strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
layer_normalization_1/mul_9Mullayer_normalization_1/mul_8:z:0.layer_normalization_1/strided_slice_7:output:0*
T0*
_output_shapes
: u
+layer_normalization_1/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_1/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_1/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:É
%layer_normalization_1/strided_slice_8StridedSlice&layer_normalization_1/Shape_2:output:04layer_normalization_1/strided_slice_8/stack:output:06layer_normalization_1/strided_slice_8/stack_1:output:06layer_normalization_1/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask`
layer_normalization_1/mul_10/xConst*
_output_shapes
: *
dtype0*
value	B :
layer_normalization_1/mul_10Mul'layer_normalization_1/mul_10/x:output:0.layer_normalization_1/strided_slice_8:output:0*
T0*
_output_shapes
: i
'layer_normalization_1/Reshape_4/shape/0Const*
_output_shapes
: *
dtype0*
value	B :i
'layer_normalization_1/Reshape_4/shape/3Const*
_output_shapes
: *
dtype0*
value	B :
%layer_normalization_1/Reshape_4/shapePack0layer_normalization_1/Reshape_4/shape/0:output:0layer_normalization_1/mul_9:z:0 layer_normalization_1/mul_10:z:00layer_normalization_1/Reshape_4/shape/3:output:0*
N*
T0*
_output_shapes
:
layer_normalization_1/Reshape_4Reshape	add_2:z:0.layer_normalization_1/Reshape_4/shape:output:0*
T0*(
_output_shapes
:NŹz
#layer_normalization_1/ones_2/packedPacklayer_normalization_1/mul_9:z:0*
N*
T0*
_output_shapes
:g
"layer_normalization_1/ones_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ľ
layer_normalization_1/ones_2Fill,layer_normalization_1/ones_2/packed:output:0+layer_normalization_1/ones_2/Const:output:0*
T0*
_output_shapes	
:N{
$layer_normalization_1/zeros_2/packedPacklayer_normalization_1/mul_9:z:0*
N*
T0*
_output_shapes
:h
#layer_normalization_1/zeros_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ¨
layer_normalization_1/zeros_2Fill-layer_normalization_1/zeros_2/packed:output:0,layer_normalization_1/zeros_2/Const:output:0*
T0*
_output_shapes	
:N`
layer_normalization_1/Const_4Const*
_output_shapes
: *
dtype0*
valueB `
layer_normalization_1/Const_5Const*
_output_shapes
: *
dtype0*
valueB 
(layer_normalization_1/FusedBatchNormV3_2FusedBatchNormV3(layer_normalization_1/Reshape_4:output:0%layer_normalization_1/ones_2:output:0&layer_normalization_1/zeros_2:output:0&layer_normalization_1/Const_4:output:0&layer_normalization_1/Const_5:output:0*
T0*
U0*H
_output_shapes6
4:NŹ:N:N:N:N:*
data_formatNCHW*
epsilon%o:Ž
layer_normalization_1/Reshape_5Reshape,layer_normalization_1/FusedBatchNormV3_2:y:0&layer_normalization_1/Shape_2:output:0*
T0*#
_output_shapes
:ddŹ
+layer_normalization_1/Cast_4/ReadVariableOpReadVariableOp2layer_normalization_1_cast_readvariableop_resource*
_output_shapes	
:Ź*
dtype0°
layer_normalization_1/mul_11Mul(layer_normalization_1/Reshape_5:output:03layer_normalization_1/Cast_4/ReadVariableOp:value:0*
T0*#
_output_shapes
:ddŹ
+layer_normalization_1/Cast_5/ReadVariableOpReadVariableOp4layer_normalization_1_cast_1_readvariableop_resource*
_output_shapes	
:Ź*
dtype0Š
layer_normalization_1/add_2AddV2 layer_normalization_1/mul_11:z:03layer_normalization_1/Cast_5/ReadVariableOp:value:0*
T0*#
_output_shapes
:ddŹj
IdentityIdentitylayer_normalization_1/add_2:z:0^NoOp*
T0*#
_output_shapes
:ddŹę
NoOpNoOp)^attention_head_2/StatefulPartitionedCall)^attention_head_3/StatefulPartitionedCall*^layer_normalization_1/Cast/ReadVariableOp,^layer_normalization_1/Cast_1/ReadVariableOp,^layer_normalization_1/Cast_2/ReadVariableOp,^layer_normalization_1/Cast_3/ReadVariableOp,^layer_normalization_1/Cast_4/ReadVariableOp,^layer_normalization_1/Cast_5/ReadVariableOp,^sequential_1/dense_2/BiasAdd/ReadVariableOp.^sequential_1/dense_2/Tensordot/ReadVariableOp,^sequential_1/dense_3/BiasAdd/ReadVariableOp.^sequential_1/dense_3/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:ddŹ:	dŹ: : : : : : : : : : : : 2T
(attention_head_2/StatefulPartitionedCall(attention_head_2/StatefulPartitionedCall2T
(attention_head_3/StatefulPartitionedCall(attention_head_3/StatefulPartitionedCall2V
)layer_normalization_1/Cast/ReadVariableOp)layer_normalization_1/Cast/ReadVariableOp2Z
+layer_normalization_1/Cast_1/ReadVariableOp+layer_normalization_1/Cast_1/ReadVariableOp2Z
+layer_normalization_1/Cast_2/ReadVariableOp+layer_normalization_1/Cast_2/ReadVariableOp2Z
+layer_normalization_1/Cast_3/ReadVariableOp+layer_normalization_1/Cast_3/ReadVariableOp2Z
+layer_normalization_1/Cast_4/ReadVariableOp+layer_normalization_1/Cast_4/ReadVariableOp2Z
+layer_normalization_1/Cast_5/ReadVariableOp+layer_normalization_1/Cast_5/ReadVariableOp2Z
+sequential_1/dense_2/BiasAdd/ReadVariableOp+sequential_1/dense_2/BiasAdd/ReadVariableOp2^
-sequential_1/dense_2/Tensordot/ReadVariableOp-sequential_1/dense_2/Tensordot/ReadVariableOp2Z
+sequential_1/dense_3/BiasAdd/ReadVariableOp+sequential_1/dense_3/BiasAdd/ReadVariableOp2^
-sequential_1/dense_3/Tensordot/ReadVariableOp-sequential_1/dense_3/Tensordot/ReadVariableOp:K G
#
_output_shapes
:ddŹ
 
_user_specified_nameinputs:QM

_output_shapes
:	dŹ
*
_user_specified_namecontext_sequence
Ö,
´
T__inference_emotion_detection_model_layer_call_and_return_conditional_losses_3832876
x_in7
$token_and_position_embedding_3832523:	dŹ8
$token_and_position_embedding_3832525:
XŹ-
transformer_block_3832647:
ŹŹ-
transformer_block_3832649:
ŹŹ-
transformer_block_3832651:
ŹŹ(
transformer_block_3832653:	Ź(
transformer_block_3832655:	Ź-
transformer_block_3832657:
ŹŹ(
transformer_block_3832659:	Ź-
transformer_block_3832661:
ŹŹ(
transformer_block_3832663:	Ź/
transformer_block_1_3832842:
ŹŹ/
transformer_block_1_3832844:
ŹŹ/
transformer_block_1_3832846:
ŹŹ*
transformer_block_1_3832848:	Ź*
transformer_block_1_3832850:	Ź/
transformer_block_1_3832852:
ŹŹ/
transformer_block_1_3832854:
ŹŹ/
transformer_block_1_3832856:
ŹŹ/
transformer_block_1_3832858:
ŹŹ*
transformer_block_1_3832860:	Ź/
transformer_block_1_3832862:
ŹŹ*
transformer_block_1_3832864:	Ź'
sequential_2_3832870:	Ź"
sequential_2_3832872:
identity˘$sequential_2/StatefulPartitionedCall˘4token_and_position_embedding/StatefulPartitionedCall˘)transformer_block/StatefulPartitionedCall˘+transformer_block_1/StatefulPartitionedCallš
4token_and_position_embedding/StatefulPartitionedCallStatefulPartitionedCallx_in$token_and_position_embedding_3832523$token_and_position_embedding_3832525*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	dŹ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *b
f]R[
Y__inference_token_and_position_embedding_layer_call_and_return_conditional_losses_3832522
)transformer_block/StatefulPartitionedCallStatefulPartitionedCall=token_and_position_embedding/StatefulPartitionedCall:output:0transformer_block_3832647transformer_block_3832649transformer_block_3832651transformer_block_3832653transformer_block_3832655transformer_block_3832657transformer_block_3832659transformer_block_3832661transformer_block_3832663*
Tin
2
*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:ddŹ*+
_read_only_resource_inputs
		*-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_transformer_block_layer_call_and_return_conditional_losses_3832646ß
dropout/PartitionedCallPartitionedCall2transformer_block/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:ddŹ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_layer_call_and_return_conditional_losses_3832671Ť
+transformer_block_1/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0=token_and_position_embedding/StatefulPartitionedCall:output:0transformer_block_1_3832842transformer_block_1_3832844transformer_block_1_3832846transformer_block_1_3832848transformer_block_1_3832850transformer_block_1_3832852transformer_block_1_3832854transformer_block_1_3832856transformer_block_1_3832858transformer_block_1_3832860transformer_block_1_3832862transformer_block_1_3832864*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:ddŹ*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_transformer_block_1_layer_call_and_return_conditional_losses_3832841P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 

ExpandDims
ExpandDims4transformer_block_1/StatefulPartitionedCall:output:0ExpandDims/dim:output:0*
T0*'
_output_shapes
:ddŹÖ
$global_max_pooling2d/PartitionedCallPartitionedCallExpandDims:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	Ź* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Z
fURS
Q__inference_global_max_pooling2d_layer_call_and_return_conditional_losses_3832389Ą
$sequential_2/StatefulPartitionedCallStatefulPartitionedCall-global_max_pooling2d/PartitionedCall:output:0sequential_2_3832870sequential_2_3832872*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sequential_2_layer_call_and_return_conditional_losses_3832417s
IdentityIdentity-sequential_2/StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:ţ
NoOpNoOp%^sequential_2/StatefulPartitionedCall5^token_and_position_embedding/StatefulPartitionedCall*^transformer_block/StatefulPartitionedCall,^transformer_block_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:˙˙˙˙˙˙˙˙˙: : : : : : : : : : : : : : : : : : : : : : : : : 2L
$sequential_2/StatefulPartitionedCall$sequential_2/StatefulPartitionedCall2l
4token_and_position_embedding/StatefulPartitionedCall4token_and_position_embedding/StatefulPartitionedCall2V
)transformer_block/StatefulPartitionedCall)transformer_block/StatefulPartitionedCall2Z
+transformer_block_1/StatefulPartitionedCall+transformer_block_1/StatefulPartitionedCall:I E
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_namex_in
Üĺ
í
__inference_call_3827682
inputs_for_keys
inputs_for_values
inputs_for_queries5
!tensordot_readvariableop_resource:
ŹŹ7
#tensordot_1_readvariableop_resource:
ŹŹ7
#tensordot_2_readvariableop_resource:
ŹŹ
identity˘Tensordot/ReadVariableOp˘Tensordot_1/ReadVariableOp˘Tensordot_2/ReadVariableOp|
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource* 
_output_shapes
:
ŹŹ*
dtype0h
Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"'  ,  z
Tensordot/ReshapeReshapeinputs_for_keys Tensordot/Reshape/shape:output:0*
T0* 
_output_shapes
:
NŹ
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0* 
_output_shapes
:
NŹd
Tensordot/shapeConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ,  x
	TensordotReshapeTensordot/MatMul:product:0Tensordot/shape:output:0*
T0*#
_output_shapes
:ddŹ
Tensordot_1/ReadVariableOpReadVariableOp#tensordot_1_readvariableop_resource* 
_output_shapes
:
ŹŹ*
dtype0j
Tensordot_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"'  ,  
Tensordot_1/ReshapeReshapeinputs_for_values"Tensordot_1/Reshape/shape:output:0*
T0* 
_output_shapes
:
NŹ
Tensordot_1/MatMulMatMulTensordot_1/Reshape:output:0"Tensordot_1/ReadVariableOp:value:0*
T0* 
_output_shapes
:
NŹf
Tensordot_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ,  ~
Tensordot_1ReshapeTensordot_1/MatMul:product:0Tensordot_1/shape:output:0*
T0*#
_output_shapes
:ddŹ
Tensordot_2/ReadVariableOpReadVariableOp#tensordot_2_readvariableop_resource* 
_output_shapes
:
ŹŹ*
dtype0~
Tensordot_2/MatMulMatMulinputs_for_queries"Tensordot_2/ReadVariableOp:value:0*
T0*
_output_shapes
:	dŹąš
attention_matrix_3/ConstConst*
_output_shapes

:dd*
dtype0*ß¸
valueÔ¸BĐ¸dd"Ŕ¸      ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙          ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙              ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                      ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                          ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                              ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                      ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                          ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                              ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                      ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                          ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                              ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                      ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                          ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                              ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                      ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                          ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                              ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                      ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                          ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                              ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                      ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                          ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                              ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                      ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                          ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                              ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                      ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                          ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                              ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                      ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                          ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                              ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                      ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                          ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                              ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                      ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                          ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                              ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                      ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                          ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                              ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                      ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                          ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                              ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                      ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                          ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                              ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                      ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                          ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                              ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                      ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                          ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                              ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                                  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                                      ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                                          ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                                              ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                                                  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                                                      ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                                                          ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                                                              ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                                                                  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                                                                      ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                                                                          ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                                                                              ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                                                                                  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                                                                                      ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                                                                                          ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                                                                                              ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                                                                                                  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                                                                                                      ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                                                                                                          ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                                                                                                              ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                                                                                                                  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                                                                                                                      ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                                                                                                                          ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                                                                                                                              ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                                                                                                                                  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                                                                                                                                      ˙  ˙  ˙                                                                                                                                                                                                                                                                                                                                                                                                          ˙  ˙                                                                                                                                                                                                                                                                                                                                                                                                              ˙                                                                                                                                                                                                                                                                                                                                                                                                                u
 attention_matrix_3/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"˙˙˙˙d   d    
attention_matrix_3/ReshapeReshape!attention_matrix_3/Const:output:0)attention_matrix_3/Reshape/shape:output:0*
T0*"
_output_shapes
:ddm
attention_matrix_3/ShapeConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ,  p
&attention_matrix_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(attention_matrix_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(attention_matrix_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:°
 attention_matrix_3/strided_sliceStridedSlice!attention_matrix_3/Shape:output:0/attention_matrix_3/strided_slice/stack:output:01attention_matrix_3/strided_slice/stack_1:output:01attention_matrix_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
#attention_matrix_3/Tile/multiples/1Const*
_output_shapes
: *
dtype0*
value	B :e
#attention_matrix_3/Tile/multiples/2Const*
_output_shapes
: *
dtype0*
value	B :Ţ
!attention_matrix_3/Tile/multiplesPack)attention_matrix_3/strided_slice:output:0,attention_matrix_3/Tile/multiples/1:output:0,attention_matrix_3/Tile/multiples/2:output:0*
N*
T0*
_output_shapes
:
attention_matrix_3/TileTile#attention_matrix_3/Reshape:output:0*attention_matrix_3/Tile/multiples:output:0*
T0*"
_output_shapes
:dddv
!attention_matrix_3/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
attention_matrix_3/transpose	TransposeTensordot:output:0*attention_matrix_3/transpose/perm:output:0*
T0*#
_output_shapes
:dŹd
attention_matrix_3/MatMulBatchMatMulV2Tensordot_2/MatMul:product:0 attention_matrix_3/transpose:y:0*
T0*"
_output_shapes
:ddd[
attention_matrix_3/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :ds
attention_matrix_3/CastCast"attention_matrix_3/Cast/x:output:0*

DstT0*

SrcT0*
_output_shapes
: ]
attention_matrix_3/SqrtSqrtattention_matrix_3/Cast:y:0*
T0*
_output_shapes
: 
attention_matrix_3/truedivRealDiv"attention_matrix_3/MatMul:output:0attention_matrix_3/Sqrt:y:0*
T0*"
_output_shapes
:dddr
attention_matrix_3/SoftmaxSoftmaxattention_matrix_3/truediv:z:0*
T0*"
_output_shapes
:ddd
MatMulBatchMatMulV2$attention_matrix_3/Softmax:softmax:0Tensordot_1:output:0*
T0*#
_output_shapes
:ddŹZ
IdentityIdentityMatMul:output:0^NoOp*
T0*#
_output_shapes
:ddŹ
NoOpNoOp^Tensordot/ReadVariableOp^Tensordot_1/ReadVariableOp^Tensordot_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:ddŹ:ddŹ:	dŹ: : : 24
Tensordot/ReadVariableOpTensordot/ReadVariableOp28
Tensordot_1/ReadVariableOpTensordot_1/ReadVariableOp28
Tensordot_2/ReadVariableOpTensordot_2/ReadVariableOp:T P
#
_output_shapes
:ddŹ
)
_user_specified_nameinputs_for_keys:VR
#
_output_shapes
:ddŹ
+
_user_specified_nameinputs_for_values:SO

_output_shapes
:	dŹ
,
_user_specified_nameinputs_for_queries
ä
Ł
.__inference_sequential_2_layer_call_fn_3832470
dense_4_input
unknown:	Ź
	unknown_0:
identity˘StatefulPartitionedCallĺ
StatefulPartitionedCallStatefulPartitionedCalldense_4_inputunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sequential_2_layer_call_and_return_conditional_losses_3832454o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:˙˙˙˙˙˙˙˙˙Ź: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ź
'
_user_specified_namedense_4_input
ą
Í
I__inference_sequential_2_layer_call_and_return_conditional_losses_3832417

inputs"
dense_4_3832411:	Ź
dense_4_3832413:
identity˘dense_4/StatefulPartitionedCallď
dense_4/StatefulPartitionedCallStatefulPartitionedCallinputsdense_4_3832411dense_4_3832413*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_4_layer_call_and_return_conditional_losses_3832410w
IdentityIdentity(dense_4/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙h
NoOpNoOp ^dense_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:˙˙˙˙˙˙˙˙˙Ź: : 2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ź
 
_user_specified_nameinputs
­
m
Q__inference_global_max_pooling2d_layer_call_and_return_conditional_losses_3832389

inputs
identityf
Max/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      m
MaxMaxinputsMax/reduction_indices:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙]
IdentityIdentityMax:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:r n
J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Íę
í
__inference_call_3827587
inputs_for_keys
inputs_for_values
inputs_for_queries5
!tensordot_readvariableop_resource:
ŹŹ7
#tensordot_1_readvariableop_resource:
ŹŹ7
#tensordot_2_readvariableop_resource:
ŹŹ
identity˘Tensordot/ReadVariableOp˘Tensordot_1/ReadVariableOp˘Tensordot_2/ReadVariableOp|
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource* 
_output_shapes
:
ŹŹ*
dtype0h
Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"'  ,  z
Tensordot/ReshapeReshapeinputs_for_keys Tensordot/Reshape/shape:output:0*
T0* 
_output_shapes
:
NŹ
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0* 
_output_shapes
:
NŹd
Tensordot/shapeConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ,  x
	TensordotReshapeTensordot/MatMul:product:0Tensordot/shape:output:0*
T0*#
_output_shapes
:ddŹ
Tensordot_1/ReadVariableOpReadVariableOp#tensordot_1_readvariableop_resource* 
_output_shapes
:
ŹŹ*
dtype0j
Tensordot_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"'  ,  
Tensordot_1/ReshapeReshapeinputs_for_values"Tensordot_1/Reshape/shape:output:0*
T0* 
_output_shapes
:
NŹ
Tensordot_1/MatMulMatMulTensordot_1/Reshape:output:0"Tensordot_1/ReadVariableOp:value:0*
T0* 
_output_shapes
:
NŹf
Tensordot_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ,  ~
Tensordot_1ReshapeTensordot_1/MatMul:product:0Tensordot_1/shape:output:0*
T0*#
_output_shapes
:ddŹ
Tensordot_2/ReadVariableOpReadVariableOp#tensordot_2_readvariableop_resource* 
_output_shapes
:
ŹŹ*
dtype0j
Tensordot_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"'  ,  
Tensordot_2/ReshapeReshapeinputs_for_queries"Tensordot_2/Reshape/shape:output:0*
T0* 
_output_shapes
:
NŹ
Tensordot_2/MatMulMatMulTensordot_2/Reshape:output:0"Tensordot_2/ReadVariableOp:value:0*
T0* 
_output_shapes
:
NŹf
Tensordot_2/shapeConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ,  ~
Tensordot_2ReshapeTensordot_2/MatMul:product:0Tensordot_2/shape:output:0*
T0*#
_output_shapes
:ddŹąš
attention_matrix_2/ConstConst*
_output_shapes

:dd*
dtype0*ß¸
valueÔ¸BĐ¸dd"Ŕ¸      ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙          ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙              ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                      ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                          ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                              ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                      ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                          ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                              ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                      ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                          ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                              ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                      ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                          ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                              ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                      ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                          ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                              ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                      ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                          ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                              ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                      ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                          ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                              ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                      ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                          ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                              ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                      ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                          ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                              ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                      ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                          ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                              ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                      ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                          ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                              ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                      ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                          ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                              ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                      ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                          ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                              ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                      ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                          ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                              ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                      ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                          ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                              ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                      ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                          ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                              ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                      ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                          ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                              ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                                  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                                      ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                                          ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                                              ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                                                  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                                                      ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                                                          ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                                                              ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                                                                  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                                                                      ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                                                                          ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                                                                              ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                                                                                  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                                                                                      ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                                                                                          ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                                                                                              ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                                                                                                  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                                                                                                      ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                                                                                                          ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                                                                                                              ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                                                                                                                  ˙  ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                                                                                                                      ˙  ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                                                                                                                          ˙  ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                                                                                                                              ˙  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                                                                                                                                  ˙  ˙  ˙  ˙                                                                                                                                                                                                                                                                                                                                                                                                      ˙  ˙  ˙                                                                                                                                                                                                                                                                                                                                                                                                          ˙  ˙                                                                                                                                                                                                                                                                                                                                                                                                              ˙                                                                                                                                                                                                                                                                                                                                                                                                                u
 attention_matrix_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"˙˙˙˙d   d    
attention_matrix_2/ReshapeReshape!attention_matrix_2/Const:output:0)attention_matrix_2/Reshape/shape:output:0*
T0*"
_output_shapes
:ddm
attention_matrix_2/ShapeConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ,  p
&attention_matrix_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(attention_matrix_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(attention_matrix_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:°
 attention_matrix_2/strided_sliceStridedSlice!attention_matrix_2/Shape:output:0/attention_matrix_2/strided_slice/stack:output:01attention_matrix_2/strided_slice/stack_1:output:01attention_matrix_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
#attention_matrix_2/Tile/multiples/1Const*
_output_shapes
: *
dtype0*
value	B :e
#attention_matrix_2/Tile/multiples/2Const*
_output_shapes
: *
dtype0*
value	B :Ţ
!attention_matrix_2/Tile/multiplesPack)attention_matrix_2/strided_slice:output:0,attention_matrix_2/Tile/multiples/1:output:0,attention_matrix_2/Tile/multiples/2:output:0*
N*
T0*
_output_shapes
:
attention_matrix_2/TileTile#attention_matrix_2/Reshape:output:0*attention_matrix_2/Tile/multiples:output:0*
T0*"
_output_shapes
:dddv
!attention_matrix_2/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
attention_matrix_2/transpose	TransposeTensordot:output:0*attention_matrix_2/transpose/perm:output:0*
T0*#
_output_shapes
:dŹd
attention_matrix_2/MatMulBatchMatMulV2Tensordot_2:output:0 attention_matrix_2/transpose:y:0*
T0*"
_output_shapes
:ddd[
attention_matrix_2/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :ds
attention_matrix_2/CastCast"attention_matrix_2/Cast/x:output:0*

DstT0*

SrcT0*
_output_shapes
: ]
attention_matrix_2/SqrtSqrtattention_matrix_2/Cast:y:0*
T0*
_output_shapes
: 
attention_matrix_2/truedivRealDiv"attention_matrix_2/MatMul:output:0attention_matrix_2/Sqrt:y:0*
T0*"
_output_shapes
:ddd
attention_matrix_2/addAddV2attention_matrix_2/truediv:z:0 attention_matrix_2/Tile:output:0*
T0*"
_output_shapes
:dddn
attention_matrix_2/SoftmaxSoftmaxattention_matrix_2/add:z:0*
T0*"
_output_shapes
:ddd
MatMulBatchMatMulV2$attention_matrix_2/Softmax:softmax:0Tensordot_1:output:0*
T0*#
_output_shapes
:ddŹZ
IdentityIdentityMatMul:output:0^NoOp*
T0*#
_output_shapes
:ddŹ
NoOpNoOp^Tensordot/ReadVariableOp^Tensordot_1/ReadVariableOp^Tensordot_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:ddŹ:ddŹ:ddŹ: : : 24
Tensordot/ReadVariableOpTensordot/ReadVariableOp28
Tensordot_1/ReadVariableOpTensordot_1/ReadVariableOp28
Tensordot_2/ReadVariableOpTensordot_2/ReadVariableOp:T P
#
_output_shapes
:ddŹ
)
_user_specified_nameinputs_for_keys:VR
#
_output_shapes
:ddŹ
+
_user_specified_nameinputs_for_values:WS
#
_output_shapes
:ddŹ
,
_user_specified_nameinputs_for_queries
ß,
ˇ
T__inference_emotion_detection_model_layer_call_and_return_conditional_losses_3833653
input_17
$token_and_position_embedding_3833594:	dŹ8
$token_and_position_embedding_3833596:
XŹ-
transformer_block_3833599:
ŹŹ-
transformer_block_3833601:
ŹŹ-
transformer_block_3833603:
ŹŹ(
transformer_block_3833605:	Ź(
transformer_block_3833607:	Ź-
transformer_block_3833609:
ŹŹ(
transformer_block_3833611:	Ź-
transformer_block_3833613:
ŹŹ(
transformer_block_3833615:	Ź/
transformer_block_1_3833619:
ŹŹ/
transformer_block_1_3833621:
ŹŹ/
transformer_block_1_3833623:
ŹŹ*
transformer_block_1_3833625:	Ź*
transformer_block_1_3833627:	Ź/
transformer_block_1_3833629:
ŹŹ/
transformer_block_1_3833631:
ŹŹ/
transformer_block_1_3833633:
ŹŹ/
transformer_block_1_3833635:
ŹŹ*
transformer_block_1_3833637:	Ź/
transformer_block_1_3833639:
ŹŹ*
transformer_block_1_3833641:	Ź'
sequential_2_3833647:	Ź"
sequential_2_3833649:
identity˘$sequential_2/StatefulPartitionedCall˘4token_and_position_embedding/StatefulPartitionedCall˘)transformer_block/StatefulPartitionedCall˘+transformer_block_1/StatefulPartitionedCallź
4token_and_position_embedding/StatefulPartitionedCallStatefulPartitionedCallinput_1$token_and_position_embedding_3833594$token_and_position_embedding_3833596*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	dŹ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *b
f]R[
Y__inference_token_and_position_embedding_layer_call_and_return_conditional_losses_3832522
)transformer_block/StatefulPartitionedCallStatefulPartitionedCall=token_and_position_embedding/StatefulPartitionedCall:output:0transformer_block_3833599transformer_block_3833601transformer_block_3833603transformer_block_3833605transformer_block_3833607transformer_block_3833609transformer_block_3833611transformer_block_3833613transformer_block_3833615*
Tin
2
*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:ddŹ*+
_read_only_resource_inputs
		*-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_transformer_block_layer_call_and_return_conditional_losses_3832646ß
dropout/PartitionedCallPartitionedCall2transformer_block/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:ddŹ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_layer_call_and_return_conditional_losses_3832671Ť
+transformer_block_1/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0=token_and_position_embedding/StatefulPartitionedCall:output:0transformer_block_1_3833619transformer_block_1_3833621transformer_block_1_3833623transformer_block_1_3833625transformer_block_1_3833627transformer_block_1_3833629transformer_block_1_3833631transformer_block_1_3833633transformer_block_1_3833635transformer_block_1_3833637transformer_block_1_3833639transformer_block_1_3833641*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:ddŹ*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_transformer_block_1_layer_call_and_return_conditional_losses_3832841P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 

ExpandDims
ExpandDims4transformer_block_1/StatefulPartitionedCall:output:0ExpandDims/dim:output:0*
T0*'
_output_shapes
:ddŹÖ
$global_max_pooling2d/PartitionedCallPartitionedCallExpandDims:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	Ź* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Z
fURS
Q__inference_global_max_pooling2d_layer_call_and_return_conditional_losses_3832389Ą
$sequential_2/StatefulPartitionedCallStatefulPartitionedCall-global_max_pooling2d/PartitionedCall:output:0sequential_2_3833647sequential_2_3833649*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sequential_2_layer_call_and_return_conditional_losses_3832417s
IdentityIdentity-sequential_2/StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:ţ
NoOpNoOp%^sequential_2/StatefulPartitionedCall5^token_and_position_embedding/StatefulPartitionedCall*^transformer_block/StatefulPartitionedCall,^transformer_block_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:˙˙˙˙˙˙˙˙˙: : : : : : : : : : : : : : : : : : : : : : : : : 2L
$sequential_2/StatefulPartitionedCall$sequential_2/StatefulPartitionedCall2l
4token_and_position_embedding/StatefulPartitionedCall4token_and_position_embedding/StatefulPartitionedCall2V
)transformer_block/StatefulPartitionedCall)transformer_block/StatefulPartitionedCall2Z
+transformer_block_1/StatefulPartitionedCall+transformer_block_1/StatefulPartitionedCall:L H
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
!
_user_specified_name	input_1
¸
ţ
D__inference_dense_2_layer_call_and_return_conditional_losses_3835258

inputs5
!tensordot_readvariableop_resource:
ŹŹ.
biasadd_readvariableop_resource:	Ź
identity˘BiasAdd/ReadVariableOp˘Tensordot/ReadVariableOp|
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource* 
_output_shapes
:
ŹŹ*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ť
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ż
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:z
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙dŹ
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ź\
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:ŹY
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : §
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙dŹs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:Ź*
dtype0}
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙dŹV
	LeakyRelu	LeakyReluBiasAdd:output:0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙dŹk
IdentityIdentityLeakyRelu:activations:0^NoOp*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙dŹz
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:˙˙˙˙˙˙˙˙˙dŹ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:˙˙˙˙˙˙˙˙˙dŹ
 
_user_specified_nameinputs
°
ś
I__inference_sequential_1_layer_call_and_return_conditional_losses_3832267

inputs#
dense_2_3832225:
ŹŹ
dense_2_3832227:	Ź#
dense_3_3832261:
ŹŹ
dense_3_3832263:	Ź
identity˘dense_2/StatefulPartitionedCall˘dense_3/StatefulPartitionedCallô
dense_2/StatefulPartitionedCallStatefulPartitionedCallinputsdense_2_3832225dense_2_3832227*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:˙˙˙˙˙˙˙˙˙dŹ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_2_layer_call_and_return_conditional_losses_3832224
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_3832261dense_3_3832263*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:˙˙˙˙˙˙˙˙˙dŹ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_3_layer_call_and_return_conditional_losses_3832260|
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙dŹ
NoOpNoOp ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :˙˙˙˙˙˙˙˙˙dŹ: : : : 2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall:T P
,
_output_shapes
:˙˙˙˙˙˙˙˙˙dŹ
 
_user_specified_nameinputs
Ď
Ţ
.__inference_sequential_1_layer_call_fn_3832278
dense_2_input
unknown:
ŹŹ
	unknown_0:	Ź
	unknown_1:
ŹŹ
	unknown_2:	Ź
identity˘StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCalldense_2_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:˙˙˙˙˙˙˙˙˙dŹ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sequential_1_layer_call_and_return_conditional_losses_3832267t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙dŹ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :˙˙˙˙˙˙˙˙˙dŹ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
,
_output_shapes
:˙˙˙˙˙˙˙˙˙dŹ
'
_user_specified_namedense_2_input
ß
b
)__inference_dropout_layer_call_fn_3834822

inputs
identity˘StatefulPartitionedCallť
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:ddŹ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_layer_call_and_return_conditional_losses_3833178k
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*#
_output_shapes
:ddŹ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*"
_input_shapes
:ddŹ22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:ddŹ
 
_user_specified_nameinputs
ł

Y__inference_token_and_position_embedding_layer_call_and_return_conditional_losses_3832522
x7
$embedding_1_embedding_lookup_3832502:	dŹ%
embedding_3832517:
XŹ
identity˘!embedding/StatefulPartitionedCall˘embedding_1/embedding_lookupM
range/startConst*
_output_shapes
: *
dtype0*
value	B : M
range/limitConst*
_output_shapes
: *
dtype0*
value	B :dM
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :l
rangeRangerange/start:output:0range/limit:output:0range/delta:output:0*
_output_shapes
:dŰ
embedding_1/embedding_lookupResourceGather$embedding_1_embedding_lookup_3832502range:output:0*
Tindices0*7
_class-
+)loc:@embedding_1/embedding_lookup/3832502*
_output_shapes
:	dŹ*
dtype0ť
%embedding_1/embedding_lookup/IdentityIdentity%embedding_1/embedding_lookup:output:0*
T0*7
_class-
+)loc:@embedding_1/embedding_lookup/3832502*
_output_shapes
:	dŹ
'embedding_1/embedding_lookup/Identity_1Identity.embedding_1/embedding_lookup/Identity:output:0*
T0*
_output_shapes
:	dŹŢ
!embedding/StatefulPartitionedCallStatefulPartitionedCallxembedding_3832517*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ź*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_embedding_layer_call_and_return_conditional_losses_3832516
addAddV2*embedding/StatefulPartitionedCall:output:00embedding_1/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:	dŹN
IdentityIdentityadd:z:0^NoOp*
T0*
_output_shapes
:	dŹ
NoOpNoOp"^embedding/StatefulPartitionedCall^embedding_1/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:˙˙˙˙˙˙˙˙˙: : 2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall2<
embedding_1/embedding_lookupembedding_1/embedding_lookup:F B
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_namex
łŔ
ç

P__inference_transformer_block_1_layer_call_and_return_conditional_losses_3832841

inputs
context_sequence,
attention_head_2_3832677:
ŹŹ,
attention_head_2_3832679:
ŹŹ,
attention_head_2_3832681:
ŹŹA
2layer_normalization_1_cast_readvariableop_resource:	ŹC
4layer_normalization_1_cast_1_readvariableop_resource:	Ź,
attention_head_3_3832728:
ŹŹ,
attention_head_3_3832730:
ŹŹ,
attention_head_3_3832732:
ŹŹJ
6sequential_1_dense_2_tensordot_readvariableop_resource:
ŹŹC
4sequential_1_dense_2_biasadd_readvariableop_resource:	ŹJ
6sequential_1_dense_3_tensordot_readvariableop_resource:
ŹŹC
4sequential_1_dense_3_biasadd_readvariableop_resource:	Ź
identity˘(attention_head_2/StatefulPartitionedCall˘(attention_head_3/StatefulPartitionedCall˘)layer_normalization_1/Cast/ReadVariableOp˘+layer_normalization_1/Cast_1/ReadVariableOp˘+layer_normalization_1/Cast_2/ReadVariableOp˘+layer_normalization_1/Cast_3/ReadVariableOp˘+layer_normalization_1/Cast_4/ReadVariableOp˘+layer_normalization_1/Cast_5/ReadVariableOp˘+sequential_1/dense_2/BiasAdd/ReadVariableOp˘-sequential_1/dense_2/Tensordot/ReadVariableOp˘+sequential_1/dense_3/BiasAdd/ReadVariableOp˘-sequential_1/dense_3/Tensordot/ReadVariableOp
(attention_head_2/StatefulPartitionedCallStatefulPartitionedCallinputsinputsinputsattention_head_2_3832677attention_head_2_3832679attention_head_2_3832681*
Tin

2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:ddŹ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *!
fR
__inference_call_3827587u
addAddV21attention_head_2/StatefulPartitionedCall:output:0inputs*
T0*#
_output_shapes
:ddŹp
layer_normalization_1/ShapeConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ,  s
)layer_normalization_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+layer_normalization_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+layer_normalization_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ż
#layer_normalization_1/strided_sliceStridedSlice$layer_normalization_1/Shape:output:02layer_normalization_1/strided_slice/stack:output:04layer_normalization_1/strided_slice/stack_1:output:04layer_normalization_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
layer_normalization_1/mul/xConst*
_output_shapes
: *
dtype0*
value	B :
layer_normalization_1/mulMul$layer_normalization_1/mul/x:output:0,layer_normalization_1/strided_slice:output:0*
T0*
_output_shapes
: u
+layer_normalization_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ç
%layer_normalization_1/strided_slice_1StridedSlice$layer_normalization_1/Shape:output:04layer_normalization_1/strided_slice_1/stack:output:06layer_normalization_1/strided_slice_1/stack_1:output:06layer_normalization_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
layer_normalization_1/mul_1Mullayer_normalization_1/mul:z:0.layer_normalization_1/strided_slice_1:output:0*
T0*
_output_shapes
: u
+layer_normalization_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ç
%layer_normalization_1/strided_slice_2StridedSlice$layer_normalization_1/Shape:output:04layer_normalization_1/strided_slice_2/stack:output:06layer_normalization_1/strided_slice_2/stack_1:output:06layer_normalization_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
layer_normalization_1/mul_2/xConst*
_output_shapes
: *
dtype0*
value	B :
layer_normalization_1/mul_2Mul&layer_normalization_1/mul_2/x:output:0.layer_normalization_1/strided_slice_2:output:0*
T0*
_output_shapes
: g
%layer_normalization_1/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :g
%layer_normalization_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :ű
#layer_normalization_1/Reshape/shapePack.layer_normalization_1/Reshape/shape/0:output:0layer_normalization_1/mul_1:z:0layer_normalization_1/mul_2:z:0.layer_normalization_1/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:
layer_normalization_1/ReshapeReshapeadd:z:0,layer_normalization_1/Reshape/shape:output:0*
T0*(
_output_shapes
:NŹx
!layer_normalization_1/ones/packedPacklayer_normalization_1/mul_1:z:0*
N*
T0*
_output_shapes
:e
 layer_normalization_1/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
layer_normalization_1/onesFill*layer_normalization_1/ones/packed:output:0)layer_normalization_1/ones/Const:output:0*
T0*
_output_shapes	
:Ny
"layer_normalization_1/zeros/packedPacklayer_normalization_1/mul_1:z:0*
N*
T0*
_output_shapes
:f
!layer_normalization_1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ˘
layer_normalization_1/zerosFill+layer_normalization_1/zeros/packed:output:0*layer_normalization_1/zeros/Const:output:0*
T0*
_output_shapes	
:N^
layer_normalization_1/ConstConst*
_output_shapes
: *
dtype0*
valueB `
layer_normalization_1/Const_1Const*
_output_shapes
: *
dtype0*
valueB 
&layer_normalization_1/FusedBatchNormV3FusedBatchNormV3&layer_normalization_1/Reshape:output:0#layer_normalization_1/ones:output:0$layer_normalization_1/zeros:output:0$layer_normalization_1/Const:output:0&layer_normalization_1/Const_1:output:0*
T0*
U0*H
_output_shapes6
4:NŹ:N:N:N:N:*
data_formatNCHW*
epsilon%o:Ş
layer_normalization_1/Reshape_1Reshape*layer_normalization_1/FusedBatchNormV3:y:0$layer_normalization_1/Shape:output:0*
T0*#
_output_shapes
:ddŹ
)layer_normalization_1/Cast/ReadVariableOpReadVariableOp2layer_normalization_1_cast_readvariableop_resource*
_output_shapes	
:Ź*
dtype0­
layer_normalization_1/mul_3Mul(layer_normalization_1/Reshape_1:output:01layer_normalization_1/Cast/ReadVariableOp:value:0*
T0*#
_output_shapes
:ddŹ
+layer_normalization_1/Cast_1/ReadVariableOpReadVariableOp4layer_normalization_1_cast_1_readvariableop_resource*
_output_shapes	
:Ź*
dtype0Ś
layer_normalization_1/addAddV2layer_normalization_1/mul_3:z:03layer_normalization_1/Cast_1/ReadVariableOp:value:0*
T0*#
_output_shapes
:ddŹŔ
(attention_head_3/StatefulPartitionedCallStatefulPartitionedCalllayer_normalization_1/add:z:0layer_normalization_1/add:z:0context_sequenceattention_head_3_3832728attention_head_3_3832730attention_head_3_3832732*
Tin

2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:ddŹ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *!
fR
__inference_call_3827682
add_1AddV21attention_head_3/StatefulPartitionedCall:output:0layer_normalization_1/add:z:0*
T0*#
_output_shapes
:ddŹr
layer_normalization_1/Shape_1Const*
_output_shapes
:*
dtype0*!
valueB"d   d   ,  u
+layer_normalization_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-layer_normalization_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:É
%layer_normalization_1/strided_slice_3StridedSlice&layer_normalization_1/Shape_1:output:04layer_normalization_1/strided_slice_3/stack:output:06layer_normalization_1/strided_slice_3/stack_1:output:06layer_normalization_1/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
layer_normalization_1/mul_4/xConst*
_output_shapes
: *
dtype0*
value	B :
layer_normalization_1/mul_4Mul&layer_normalization_1/mul_4/x:output:0.layer_normalization_1/strided_slice_3:output:0*
T0*
_output_shapes
: u
+layer_normalization_1/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_1/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_1/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:É
%layer_normalization_1/strided_slice_4StridedSlice&layer_normalization_1/Shape_1:output:04layer_normalization_1/strided_slice_4/stack:output:06layer_normalization_1/strided_slice_4/stack_1:output:06layer_normalization_1/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
layer_normalization_1/mul_5Mullayer_normalization_1/mul_4:z:0.layer_normalization_1/strided_slice_4:output:0*
T0*
_output_shapes
: u
+layer_normalization_1/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_1/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_1/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:É
%layer_normalization_1/strided_slice_5StridedSlice&layer_normalization_1/Shape_1:output:04layer_normalization_1/strided_slice_5/stack:output:06layer_normalization_1/strided_slice_5/stack_1:output:06layer_normalization_1/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
layer_normalization_1/mul_6/xConst*
_output_shapes
: *
dtype0*
value	B :
layer_normalization_1/mul_6Mul&layer_normalization_1/mul_6/x:output:0.layer_normalization_1/strided_slice_5:output:0*
T0*
_output_shapes
: i
'layer_normalization_1/Reshape_2/shape/0Const*
_output_shapes
: *
dtype0*
value	B :i
'layer_normalization_1/Reshape_2/shape/3Const*
_output_shapes
: *
dtype0*
value	B :
%layer_normalization_1/Reshape_2/shapePack0layer_normalization_1/Reshape_2/shape/0:output:0layer_normalization_1/mul_5:z:0layer_normalization_1/mul_6:z:00layer_normalization_1/Reshape_2/shape/3:output:0*
N*
T0*
_output_shapes
:
layer_normalization_1/Reshape_2Reshape	add_1:z:0.layer_normalization_1/Reshape_2/shape:output:0*
T0*(
_output_shapes
:NŹz
#layer_normalization_1/ones_1/packedPacklayer_normalization_1/mul_5:z:0*
N*
T0*
_output_shapes
:g
"layer_normalization_1/ones_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ľ
layer_normalization_1/ones_1Fill,layer_normalization_1/ones_1/packed:output:0+layer_normalization_1/ones_1/Const:output:0*
T0*
_output_shapes	
:N{
$layer_normalization_1/zeros_1/packedPacklayer_normalization_1/mul_5:z:0*
N*
T0*
_output_shapes
:h
#layer_normalization_1/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ¨
layer_normalization_1/zeros_1Fill-layer_normalization_1/zeros_1/packed:output:0,layer_normalization_1/zeros_1/Const:output:0*
T0*
_output_shapes	
:N`
layer_normalization_1/Const_2Const*
_output_shapes
: *
dtype0*
valueB `
layer_normalization_1/Const_3Const*
_output_shapes
: *
dtype0*
valueB 
(layer_normalization_1/FusedBatchNormV3_1FusedBatchNormV3(layer_normalization_1/Reshape_2:output:0%layer_normalization_1/ones_1:output:0&layer_normalization_1/zeros_1:output:0&layer_normalization_1/Const_2:output:0&layer_normalization_1/Const_3:output:0*
T0*
U0*H
_output_shapes6
4:NŹ:N:N:N:N:*
data_formatNCHW*
epsilon%o:Ž
layer_normalization_1/Reshape_3Reshape,layer_normalization_1/FusedBatchNormV3_1:y:0&layer_normalization_1/Shape_1:output:0*
T0*#
_output_shapes
:ddŹ
+layer_normalization_1/Cast_2/ReadVariableOpReadVariableOp2layer_normalization_1_cast_readvariableop_resource*
_output_shapes	
:Ź*
dtype0Ż
layer_normalization_1/mul_7Mul(layer_normalization_1/Reshape_3:output:03layer_normalization_1/Cast_2/ReadVariableOp:value:0*
T0*#
_output_shapes
:ddŹ
+layer_normalization_1/Cast_3/ReadVariableOpReadVariableOp4layer_normalization_1_cast_1_readvariableop_resource*
_output_shapes	
:Ź*
dtype0¨
layer_normalization_1/add_1AddV2layer_normalization_1/mul_7:z:03layer_normalization_1/Cast_3/ReadVariableOp:value:0*
T0*#
_output_shapes
:ddŹŚ
-sequential_1/dense_2/Tensordot/ReadVariableOpReadVariableOp6sequential_1_dense_2_tensordot_readvariableop_resource* 
_output_shapes
:
ŹŹ*
dtype0}
,sequential_1/dense_2/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"'  ,  ´
&sequential_1/dense_2/Tensordot/ReshapeReshapelayer_normalization_1/add_1:z:05sequential_1/dense_2/Tensordot/Reshape/shape:output:0*
T0* 
_output_shapes
:
NŹÂ
%sequential_1/dense_2/Tensordot/MatMulMatMul/sequential_1/dense_2/Tensordot/Reshape:output:05sequential_1/dense_2/Tensordot/ReadVariableOp:value:0*
T0* 
_output_shapes
:
NŹy
$sequential_1/dense_2/Tensordot/shapeConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ,  ˇ
sequential_1/dense_2/TensordotReshape/sequential_1/dense_2/Tensordot/MatMul:product:0-sequential_1/dense_2/Tensordot/shape:output:0*
T0*#
_output_shapes
:ddŹ
+sequential_1/dense_2/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:Ź*
dtype0ł
sequential_1/dense_2/BiasAddBiasAdd'sequential_1/dense_2/Tensordot:output:03sequential_1/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*#
_output_shapes
:ddŹw
sequential_1/dense_2/LeakyRelu	LeakyRelu%sequential_1/dense_2/BiasAdd:output:0*#
_output_shapes
:ddŹŚ
-sequential_1/dense_3/Tensordot/ReadVariableOpReadVariableOp6sequential_1_dense_3_tensordot_readvariableop_resource* 
_output_shapes
:
ŹŹ*
dtype0}
,sequential_1/dense_3/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"'  ,  Á
&sequential_1/dense_3/Tensordot/ReshapeReshape,sequential_1/dense_2/LeakyRelu:activations:05sequential_1/dense_3/Tensordot/Reshape/shape:output:0*
T0* 
_output_shapes
:
NŹÂ
%sequential_1/dense_3/Tensordot/MatMulMatMul/sequential_1/dense_3/Tensordot/Reshape:output:05sequential_1/dense_3/Tensordot/ReadVariableOp:value:0*
T0* 
_output_shapes
:
NŹy
$sequential_1/dense_3/Tensordot/shapeConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ,  ˇ
sequential_1/dense_3/TensordotReshape/sequential_1/dense_3/Tensordot/MatMul:product:0-sequential_1/dense_3/Tensordot/shape:output:0*
T0*#
_output_shapes
:ddŹ
+sequential_1/dense_3/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_3_biasadd_readvariableop_resource*
_output_shapes	
:Ź*
dtype0ł
sequential_1/dense_3/BiasAddBiasAdd'sequential_1/dense_3/Tensordot:output:03sequential_1/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*#
_output_shapes
:ddŹ
add_2AddV2%sequential_1/dense_3/BiasAdd:output:0layer_normalization_1/add_1:z:0*
T0*#
_output_shapes
:ddŹr
layer_normalization_1/Shape_2Const*
_output_shapes
:*
dtype0*!
valueB"d   d   ,  u
+layer_normalization_1/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-layer_normalization_1/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_1/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB:É
%layer_normalization_1/strided_slice_6StridedSlice&layer_normalization_1/Shape_2:output:04layer_normalization_1/strided_slice_6/stack:output:06layer_normalization_1/strided_slice_6/stack_1:output:06layer_normalization_1/strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
layer_normalization_1/mul_8/xConst*
_output_shapes
: *
dtype0*
value	B :
layer_normalization_1/mul_8Mul&layer_normalization_1/mul_8/x:output:0.layer_normalization_1/strided_slice_6:output:0*
T0*
_output_shapes
: u
+layer_normalization_1/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_1/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_1/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB:É
%layer_normalization_1/strided_slice_7StridedSlice&layer_normalization_1/Shape_2:output:04layer_normalization_1/strided_slice_7/stack:output:06layer_normalization_1/strided_slice_7/stack_1:output:06layer_normalization_1/strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
layer_normalization_1/mul_9Mullayer_normalization_1/mul_8:z:0.layer_normalization_1/strided_slice_7:output:0*
T0*
_output_shapes
: u
+layer_normalization_1/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_1/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_1/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:É
%layer_normalization_1/strided_slice_8StridedSlice&layer_normalization_1/Shape_2:output:04layer_normalization_1/strided_slice_8/stack:output:06layer_normalization_1/strided_slice_8/stack_1:output:06layer_normalization_1/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask`
layer_normalization_1/mul_10/xConst*
_output_shapes
: *
dtype0*
value	B :
layer_normalization_1/mul_10Mul'layer_normalization_1/mul_10/x:output:0.layer_normalization_1/strided_slice_8:output:0*
T0*
_output_shapes
: i
'layer_normalization_1/Reshape_4/shape/0Const*
_output_shapes
: *
dtype0*
value	B :i
'layer_normalization_1/Reshape_4/shape/3Const*
_output_shapes
: *
dtype0*
value	B :
%layer_normalization_1/Reshape_4/shapePack0layer_normalization_1/Reshape_4/shape/0:output:0layer_normalization_1/mul_9:z:0 layer_normalization_1/mul_10:z:00layer_normalization_1/Reshape_4/shape/3:output:0*
N*
T0*
_output_shapes
:
layer_normalization_1/Reshape_4Reshape	add_2:z:0.layer_normalization_1/Reshape_4/shape:output:0*
T0*(
_output_shapes
:NŹz
#layer_normalization_1/ones_2/packedPacklayer_normalization_1/mul_9:z:0*
N*
T0*
_output_shapes
:g
"layer_normalization_1/ones_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ľ
layer_normalization_1/ones_2Fill,layer_normalization_1/ones_2/packed:output:0+layer_normalization_1/ones_2/Const:output:0*
T0*
_output_shapes	
:N{
$layer_normalization_1/zeros_2/packedPacklayer_normalization_1/mul_9:z:0*
N*
T0*
_output_shapes
:h
#layer_normalization_1/zeros_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ¨
layer_normalization_1/zeros_2Fill-layer_normalization_1/zeros_2/packed:output:0,layer_normalization_1/zeros_2/Const:output:0*
T0*
_output_shapes	
:N`
layer_normalization_1/Const_4Const*
_output_shapes
: *
dtype0*
valueB `
layer_normalization_1/Const_5Const*
_output_shapes
: *
dtype0*
valueB 
(layer_normalization_1/FusedBatchNormV3_2FusedBatchNormV3(layer_normalization_1/Reshape_4:output:0%layer_normalization_1/ones_2:output:0&layer_normalization_1/zeros_2:output:0&layer_normalization_1/Const_4:output:0&layer_normalization_1/Const_5:output:0*
T0*
U0*H
_output_shapes6
4:NŹ:N:N:N:N:*
data_formatNCHW*
epsilon%o:Ž
layer_normalization_1/Reshape_5Reshape,layer_normalization_1/FusedBatchNormV3_2:y:0&layer_normalization_1/Shape_2:output:0*
T0*#
_output_shapes
:ddŹ
+layer_normalization_1/Cast_4/ReadVariableOpReadVariableOp2layer_normalization_1_cast_readvariableop_resource*
_output_shapes	
:Ź*
dtype0°
layer_normalization_1/mul_11Mul(layer_normalization_1/Reshape_5:output:03layer_normalization_1/Cast_4/ReadVariableOp:value:0*
T0*#
_output_shapes
:ddŹ
+layer_normalization_1/Cast_5/ReadVariableOpReadVariableOp4layer_normalization_1_cast_1_readvariableop_resource*
_output_shapes	
:Ź*
dtype0Š
layer_normalization_1/add_2AddV2 layer_normalization_1/mul_11:z:03layer_normalization_1/Cast_5/ReadVariableOp:value:0*
T0*#
_output_shapes
:ddŹj
IdentityIdentitylayer_normalization_1/add_2:z:0^NoOp*
T0*#
_output_shapes
:ddŹę
NoOpNoOp)^attention_head_2/StatefulPartitionedCall)^attention_head_3/StatefulPartitionedCall*^layer_normalization_1/Cast/ReadVariableOp,^layer_normalization_1/Cast_1/ReadVariableOp,^layer_normalization_1/Cast_2/ReadVariableOp,^layer_normalization_1/Cast_3/ReadVariableOp,^layer_normalization_1/Cast_4/ReadVariableOp,^layer_normalization_1/Cast_5/ReadVariableOp,^sequential_1/dense_2/BiasAdd/ReadVariableOp.^sequential_1/dense_2/Tensordot/ReadVariableOp,^sequential_1/dense_3/BiasAdd/ReadVariableOp.^sequential_1/dense_3/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:ddŹ:	dŹ: : : : : : : : : : : : 2T
(attention_head_2/StatefulPartitionedCall(attention_head_2/StatefulPartitionedCall2T
(attention_head_3/StatefulPartitionedCall(attention_head_3/StatefulPartitionedCall2V
)layer_normalization_1/Cast/ReadVariableOp)layer_normalization_1/Cast/ReadVariableOp2Z
+layer_normalization_1/Cast_1/ReadVariableOp+layer_normalization_1/Cast_1/ReadVariableOp2Z
+layer_normalization_1/Cast_2/ReadVariableOp+layer_normalization_1/Cast_2/ReadVariableOp2Z
+layer_normalization_1/Cast_3/ReadVariableOp+layer_normalization_1/Cast_3/ReadVariableOp2Z
+layer_normalization_1/Cast_4/ReadVariableOp+layer_normalization_1/Cast_4/ReadVariableOp2Z
+layer_normalization_1/Cast_5/ReadVariableOp+layer_normalization_1/Cast_5/ReadVariableOp2Z
+sequential_1/dense_2/BiasAdd/ReadVariableOp+sequential_1/dense_2/BiasAdd/ReadVariableOp2^
-sequential_1/dense_2/Tensordot/ReadVariableOp-sequential_1/dense_2/Tensordot/ReadVariableOp2Z
+sequential_1/dense_3/BiasAdd/ReadVariableOp+sequential_1/dense_3/BiasAdd/ReadVariableOp2^
-sequential_1/dense_3/Tensordot/ReadVariableOp-sequential_1/dense_3/Tensordot/ReadVariableOp:K G
#
_output_shapes
:ddŹ
 
_user_specified_nameinputs:QM

_output_shapes
:	dŹ
*
_user_specified_namecontext_sequence
ń
Ü
__inference_call_3830033

inputs*
attention_head_3829918:
ŹŹ*
attention_head_3829920:
ŹŹ*
attention_head_3829922:
ŹŹ?
0layer_normalization_cast_readvariableop_resource:	ŹA
2layer_normalization_cast_1_readvariableop_resource:	ŹF
2sequential_dense_tensordot_readvariableop_resource:
ŹŹ?
0sequential_dense_biasadd_readvariableop_resource:	ŹH
4sequential_dense_1_tensordot_readvariableop_resource:
ŹŹA
2sequential_dense_1_biasadd_readvariableop_resource:	Ź
identity˘&attention_head/StatefulPartitionedCall˘'layer_normalization/Cast/ReadVariableOp˘)layer_normalization/Cast_1/ReadVariableOp˘)layer_normalization/Cast_2/ReadVariableOp˘)layer_normalization/Cast_3/ReadVariableOp˘'sequential/dense/BiasAdd/ReadVariableOp˘)sequential/dense/Tensordot/ReadVariableOp˘)sequential/dense_1/BiasAdd/ReadVariableOp˘+sequential/dense_1/Tensordot/ReadVariableOp
&attention_head/StatefulPartitionedCallStatefulPartitionedCallinputsinputsinputsattention_head_3829918attention_head_3829920attention_head_3829922*
Tin

2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:ddŹ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *!
fR
__inference_call_3827399s
addAddV2/attention_head/StatefulPartitionedCall:output:0inputs*
T0*#
_output_shapes
:ddŹn
layer_normalization/ShapeConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ,  q
'layer_normalization/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)layer_normalization/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)layer_normalization/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ľ
!layer_normalization/strided_sliceStridedSlice"layer_normalization/Shape:output:00layer_normalization/strided_slice/stack:output:02layer_normalization/strided_slice/stack_1:output:02layer_normalization/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask[
layer_normalization/mul/xConst*
_output_shapes
: *
dtype0*
value	B :
layer_normalization/mulMul"layer_normalization/mul/x:output:0*layer_normalization/strided_slice:output:0*
T0*
_output_shapes
: s
)layer_normalization/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:u
+layer_normalization/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+layer_normalization/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:˝
#layer_normalization/strided_slice_1StridedSlice"layer_normalization/Shape:output:02layer_normalization/strided_slice_1/stack:output:04layer_normalization/strided_slice_1/stack_1:output:04layer_normalization/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
layer_normalization/mul_1Mullayer_normalization/mul:z:0,layer_normalization/strided_slice_1:output:0*
T0*
_output_shapes
: s
)layer_normalization/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:u
+layer_normalization/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+layer_normalization/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:˝
#layer_normalization/strided_slice_2StridedSlice"layer_normalization/Shape:output:02layer_normalization/strided_slice_2/stack:output:04layer_normalization/strided_slice_2/stack_1:output:04layer_normalization/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
layer_normalization/mul_2/xConst*
_output_shapes
: *
dtype0*
value	B :
layer_normalization/mul_2Mul$layer_normalization/mul_2/x:output:0,layer_normalization/strided_slice_2:output:0*
T0*
_output_shapes
: e
#layer_normalization/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :e
#layer_normalization/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :ń
!layer_normalization/Reshape/shapePack,layer_normalization/Reshape/shape/0:output:0layer_normalization/mul_1:z:0layer_normalization/mul_2:z:0,layer_normalization/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:
layer_normalization/ReshapeReshapeadd:z:0*layer_normalization/Reshape/shape:output:0*
T0*(
_output_shapes
:NŹt
layer_normalization/ones/packedPacklayer_normalization/mul_1:z:0*
N*
T0*
_output_shapes
:c
layer_normalization/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
layer_normalization/onesFill(layer_normalization/ones/packed:output:0'layer_normalization/ones/Const:output:0*
T0*
_output_shapes	
:Nu
 layer_normalization/zeros/packedPacklayer_normalization/mul_1:z:0*
N*
T0*
_output_shapes
:d
layer_normalization/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
layer_normalization/zerosFill)layer_normalization/zeros/packed:output:0(layer_normalization/zeros/Const:output:0*
T0*
_output_shapes	
:N\
layer_normalization/ConstConst*
_output_shapes
: *
dtype0*
valueB ^
layer_normalization/Const_1Const*
_output_shapes
: *
dtype0*
valueB ô
$layer_normalization/FusedBatchNormV3FusedBatchNormV3$layer_normalization/Reshape:output:0!layer_normalization/ones:output:0"layer_normalization/zeros:output:0"layer_normalization/Const:output:0$layer_normalization/Const_1:output:0*
T0*
U0*H
_output_shapes6
4:NŹ:N:N:N:N:*
data_formatNCHW*
epsilon%o:¤
layer_normalization/Reshape_1Reshape(layer_normalization/FusedBatchNormV3:y:0"layer_normalization/Shape:output:0*
T0*#
_output_shapes
:ddŹ
'layer_normalization/Cast/ReadVariableOpReadVariableOp0layer_normalization_cast_readvariableop_resource*
_output_shapes	
:Ź*
dtype0§
layer_normalization/mul_3Mul&layer_normalization/Reshape_1:output:0/layer_normalization/Cast/ReadVariableOp:value:0*
T0*#
_output_shapes
:ddŹ
)layer_normalization/Cast_1/ReadVariableOpReadVariableOp2layer_normalization_cast_1_readvariableop_resource*
_output_shapes	
:Ź*
dtype0 
layer_normalization/addAddV2layer_normalization/mul_3:z:01layer_normalization/Cast_1/ReadVariableOp:value:0*
T0*#
_output_shapes
:ddŹ
)sequential/dense/Tensordot/ReadVariableOpReadVariableOp2sequential_dense_tensordot_readvariableop_resource* 
_output_shapes
:
ŹŹ*
dtype0y
(sequential/dense/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"'  ,  ¨
"sequential/dense/Tensordot/ReshapeReshapelayer_normalization/add:z:01sequential/dense/Tensordot/Reshape/shape:output:0*
T0* 
_output_shapes
:
NŹś
!sequential/dense/Tensordot/MatMulMatMul+sequential/dense/Tensordot/Reshape:output:01sequential/dense/Tensordot/ReadVariableOp:value:0*
T0* 
_output_shapes
:
NŹu
 sequential/dense/Tensordot/shapeConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ,  Ť
sequential/dense/TensordotReshape+sequential/dense/Tensordot/MatMul:product:0)sequential/dense/Tensordot/shape:output:0*
T0*#
_output_shapes
:ddŹ
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes	
:Ź*
dtype0§
sequential/dense/BiasAddBiasAdd#sequential/dense/Tensordot:output:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*#
_output_shapes
:ddŹo
sequential/dense/LeakyRelu	LeakyRelu!sequential/dense/BiasAdd:output:0*#
_output_shapes
:ddŹ˘
+sequential/dense_1/Tensordot/ReadVariableOpReadVariableOp4sequential_dense_1_tensordot_readvariableop_resource* 
_output_shapes
:
ŹŹ*
dtype0{
*sequential/dense_1/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"'  ,  š
$sequential/dense_1/Tensordot/ReshapeReshape(sequential/dense/LeakyRelu:activations:03sequential/dense_1/Tensordot/Reshape/shape:output:0*
T0* 
_output_shapes
:
NŹź
#sequential/dense_1/Tensordot/MatMulMatMul-sequential/dense_1/Tensordot/Reshape:output:03sequential/dense_1/Tensordot/ReadVariableOp:value:0*
T0* 
_output_shapes
:
NŹw
"sequential/dense_1/Tensordot/shapeConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ,  ą
sequential/dense_1/TensordotReshape-sequential/dense_1/Tensordot/MatMul:product:0+sequential/dense_1/Tensordot/shape:output:0*
T0*#
_output_shapes
:ddŹ
)sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:Ź*
dtype0­
sequential/dense_1/BiasAddBiasAdd%sequential/dense_1/Tensordot:output:01sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*#
_output_shapes
:ddŹ~
add_1AddV2#sequential/dense_1/BiasAdd:output:0layer_normalization/add:z:0*
T0*#
_output_shapes
:ddŹp
layer_normalization/Shape_1Const*
_output_shapes
:*
dtype0*!
valueB"d   d   ,  s
)layer_normalization/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+layer_normalization/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+layer_normalization/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ż
#layer_normalization/strided_slice_3StridedSlice$layer_normalization/Shape_1:output:02layer_normalization/strided_slice_3/stack:output:04layer_normalization/strided_slice_3/stack_1:output:04layer_normalization/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
layer_normalization/mul_4/xConst*
_output_shapes
: *
dtype0*
value	B :
layer_normalization/mul_4Mul$layer_normalization/mul_4/x:output:0,layer_normalization/strided_slice_3:output:0*
T0*
_output_shapes
: s
)layer_normalization/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:u
+layer_normalization/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+layer_normalization/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ż
#layer_normalization/strided_slice_4StridedSlice$layer_normalization/Shape_1:output:02layer_normalization/strided_slice_4/stack:output:04layer_normalization/strided_slice_4/stack_1:output:04layer_normalization/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
layer_normalization/mul_5Mullayer_normalization/mul_4:z:0,layer_normalization/strided_slice_4:output:0*
T0*
_output_shapes
: s
)layer_normalization/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:u
+layer_normalization/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+layer_normalization/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ż
#layer_normalization/strided_slice_5StridedSlice$layer_normalization/Shape_1:output:02layer_normalization/strided_slice_5/stack:output:04layer_normalization/strided_slice_5/stack_1:output:04layer_normalization/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
layer_normalization/mul_6/xConst*
_output_shapes
: *
dtype0*
value	B :
layer_normalization/mul_6Mul$layer_normalization/mul_6/x:output:0,layer_normalization/strided_slice_5:output:0*
T0*
_output_shapes
: g
%layer_normalization/Reshape_2/shape/0Const*
_output_shapes
: *
dtype0*
value	B :g
%layer_normalization/Reshape_2/shape/3Const*
_output_shapes
: *
dtype0*
value	B :÷
#layer_normalization/Reshape_2/shapePack.layer_normalization/Reshape_2/shape/0:output:0layer_normalization/mul_5:z:0layer_normalization/mul_6:z:0.layer_normalization/Reshape_2/shape/3:output:0*
N*
T0*
_output_shapes
:
layer_normalization/Reshape_2Reshape	add_1:z:0,layer_normalization/Reshape_2/shape:output:0*
T0*(
_output_shapes
:NŹv
!layer_normalization/ones_1/packedPacklayer_normalization/mul_5:z:0*
N*
T0*
_output_shapes
:e
 layer_normalization/ones_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
layer_normalization/ones_1Fill*layer_normalization/ones_1/packed:output:0)layer_normalization/ones_1/Const:output:0*
T0*
_output_shapes	
:Nw
"layer_normalization/zeros_1/packedPacklayer_normalization/mul_5:z:0*
N*
T0*
_output_shapes
:f
!layer_normalization/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ˘
layer_normalization/zeros_1Fill+layer_normalization/zeros_1/packed:output:0*layer_normalization/zeros_1/Const:output:0*
T0*
_output_shapes	
:N^
layer_normalization/Const_2Const*
_output_shapes
: *
dtype0*
valueB ^
layer_normalization/Const_3Const*
_output_shapes
: *
dtype0*
valueB ţ
&layer_normalization/FusedBatchNormV3_1FusedBatchNormV3&layer_normalization/Reshape_2:output:0#layer_normalization/ones_1:output:0$layer_normalization/zeros_1:output:0$layer_normalization/Const_2:output:0$layer_normalization/Const_3:output:0*
T0*
U0*H
_output_shapes6
4:NŹ:N:N:N:N:*
data_formatNCHW*
epsilon%o:¨
layer_normalization/Reshape_3Reshape*layer_normalization/FusedBatchNormV3_1:y:0$layer_normalization/Shape_1:output:0*
T0*#
_output_shapes
:ddŹ
)layer_normalization/Cast_2/ReadVariableOpReadVariableOp0layer_normalization_cast_readvariableop_resource*
_output_shapes	
:Ź*
dtype0Š
layer_normalization/mul_7Mul&layer_normalization/Reshape_3:output:01layer_normalization/Cast_2/ReadVariableOp:value:0*
T0*#
_output_shapes
:ddŹ
)layer_normalization/Cast_3/ReadVariableOpReadVariableOp2layer_normalization_cast_1_readvariableop_resource*
_output_shapes	
:Ź*
dtype0˘
layer_normalization/add_1AddV2layer_normalization/mul_7:z:01layer_normalization/Cast_3/ReadVariableOp:value:0*
T0*#
_output_shapes
:ddŹh
IdentityIdentitylayer_normalization/add_1:z:0^NoOp*
T0*#
_output_shapes
:ddŹÍ
NoOpNoOp'^attention_head/StatefulPartitionedCall(^layer_normalization/Cast/ReadVariableOp*^layer_normalization/Cast_1/ReadVariableOp*^layer_normalization/Cast_2/ReadVariableOp*^layer_normalization/Cast_3/ReadVariableOp(^sequential/dense/BiasAdd/ReadVariableOp*^sequential/dense/Tensordot/ReadVariableOp*^sequential/dense_1/BiasAdd/ReadVariableOp,^sequential/dense_1/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:	dŹ: : : : : : : : : 2P
&attention_head/StatefulPartitionedCall&attention_head/StatefulPartitionedCall2R
'layer_normalization/Cast/ReadVariableOp'layer_normalization/Cast/ReadVariableOp2V
)layer_normalization/Cast_1/ReadVariableOp)layer_normalization/Cast_1/ReadVariableOp2V
)layer_normalization/Cast_2/ReadVariableOp)layer_normalization/Cast_2/ReadVariableOp2V
)layer_normalization/Cast_3/ReadVariableOp)layer_normalization/Cast_3/ReadVariableOp2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2V
)sequential/dense/Tensordot/ReadVariableOp)sequential/dense/Tensordot/ReadVariableOp2V
)sequential/dense_1/BiasAdd/ReadVariableOp)sequential/dense_1/BiasAdd/ReadVariableOp2Z
+sequential/dense_1/Tensordot/ReadVariableOp+sequential/dense_1/Tensordot/ReadVariableOp:G C

_output_shapes
:	dŹ
 
_user_specified_nameinputs
ś
Ő
,__inference_sequential_layer_call_fn_3834865

inputs
unknown:
ŹŹ
	unknown_0:	Ź
	unknown_1:
ŹŹ
	unknown_2:	Ź
identity˘StatefulPartitionedCallű
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:˙˙˙˙˙˙˙˙˙dŹ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_sequential_layer_call_and_return_conditional_losses_3832134t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙dŹ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :˙˙˙˙˙˙˙˙˙dŹ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:˙˙˙˙˙˙˙˙˙dŹ
 
_user_specified_nameinputs
ö-
Ö
T__inference_emotion_detection_model_layer_call_and_return_conditional_losses_3833483
x_in7
$token_and_position_embedding_3833424:	dŹ8
$token_and_position_embedding_3833426:
XŹ-
transformer_block_3833429:
ŹŹ-
transformer_block_3833431:
ŹŹ-
transformer_block_3833433:
ŹŹ(
transformer_block_3833435:	Ź(
transformer_block_3833437:	Ź-
transformer_block_3833439:
ŹŹ(
transformer_block_3833441:	Ź-
transformer_block_3833443:
ŹŹ(
transformer_block_3833445:	Ź/
transformer_block_1_3833449:
ŹŹ/
transformer_block_1_3833451:
ŹŹ/
transformer_block_1_3833453:
ŹŹ*
transformer_block_1_3833455:	Ź*
transformer_block_1_3833457:	Ź/
transformer_block_1_3833459:
ŹŹ/
transformer_block_1_3833461:
ŹŹ/
transformer_block_1_3833463:
ŹŹ/
transformer_block_1_3833465:
ŹŹ*
transformer_block_1_3833467:	Ź/
transformer_block_1_3833469:
ŹŹ*
transformer_block_1_3833471:	Ź'
sequential_2_3833477:	Ź"
sequential_2_3833479:
identity˘dropout/StatefulPartitionedCall˘$sequential_2/StatefulPartitionedCall˘4token_and_position_embedding/StatefulPartitionedCall˘)transformer_block/StatefulPartitionedCall˘+transformer_block_1/StatefulPartitionedCallš
4token_and_position_embedding/StatefulPartitionedCallStatefulPartitionedCallx_in$token_and_position_embedding_3833424$token_and_position_embedding_3833426*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	dŹ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *b
f]R[
Y__inference_token_and_position_embedding_layer_call_and_return_conditional_losses_3832522
)transformer_block/StatefulPartitionedCallStatefulPartitionedCall=token_and_position_embedding/StatefulPartitionedCall:output:0transformer_block_3833429transformer_block_3833431transformer_block_3833433transformer_block_3833435transformer_block_3833437transformer_block_3833439transformer_block_3833441transformer_block_3833443transformer_block_3833445*
Tin
2
*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:ddŹ*+
_read_only_resource_inputs
		*-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_transformer_block_layer_call_and_return_conditional_losses_3833325ď
dropout/StatefulPartitionedCallStatefulPartitionedCall2transformer_block/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:ddŹ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_layer_call_and_return_conditional_losses_3833178ł
+transformer_block_1/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0=token_and_position_embedding/StatefulPartitionedCall:output:0transformer_block_1_3833449transformer_block_1_3833451transformer_block_1_3833453transformer_block_1_3833455transformer_block_1_3833457transformer_block_1_3833459transformer_block_1_3833461transformer_block_1_3833463transformer_block_1_3833465transformer_block_1_3833467transformer_block_1_3833469transformer_block_1_3833471*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:ddŹ*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_transformer_block_1_layer_call_and_return_conditional_losses_3833131P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 

ExpandDims
ExpandDims4transformer_block_1/StatefulPartitionedCall:output:0ExpandDims/dim:output:0*
T0*'
_output_shapes
:ddŹÖ
$global_max_pooling2d/PartitionedCallPartitionedCallExpandDims:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	Ź* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Z
fURS
Q__inference_global_max_pooling2d_layer_call_and_return_conditional_losses_3832389Ą
$sequential_2/StatefulPartitionedCallStatefulPartitionedCall-global_max_pooling2d/PartitionedCall:output:0sequential_2_3833477sequential_2_3833479*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sequential_2_layer_call_and_return_conditional_losses_3832454s
IdentityIdentity-sequential_2/StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

: 
NoOpNoOp ^dropout/StatefulPartitionedCall%^sequential_2/StatefulPartitionedCall5^token_and_position_embedding/StatefulPartitionedCall*^transformer_block/StatefulPartitionedCall,^transformer_block_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:˙˙˙˙˙˙˙˙˙: : : : : : : : : : : : : : : : : : : : : : : : : 2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2L
$sequential_2/StatefulPartitionedCall$sequential_2/StatefulPartitionedCall2l
4token_and_position_embedding/StatefulPartitionedCall4token_and_position_embedding/StatefulPartitionedCall2V
)transformer_block/StatefulPartitionedCall)transformer_block/StatefulPartitionedCall2Z
+transformer_block_1/StatefulPartitionedCall+transformer_block_1/StatefulPartitionedCall:I E
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_namex_in
G

T__inference_emotion_detection_model_layer_call_and_return_conditional_losses_3834039
x_inT
Atoken_and_position_embedding_embedding_1_embedding_lookup_3833964:	dŹS
?token_and_position_embedding_embedding_embedding_lookup_3833969:
XŹ-
transformer_block_3833975:
ŹŹ-
transformer_block_3833977:
ŹŹ-
transformer_block_3833979:
ŹŹ(
transformer_block_3833981:	Ź(
transformer_block_3833983:	Ź-
transformer_block_3833985:
ŹŹ(
transformer_block_3833987:	Ź-
transformer_block_3833989:
ŹŹ(
transformer_block_3833991:	Ź/
transformer_block_1_3834002:
ŹŹ/
transformer_block_1_3834004:
ŹŹ/
transformer_block_1_3834006:
ŹŹ*
transformer_block_1_3834008:	Ź*
transformer_block_1_3834010:	Ź/
transformer_block_1_3834012:
ŹŹ/
transformer_block_1_3834014:
ŹŹ/
transformer_block_1_3834016:
ŹŹ/
transformer_block_1_3834018:
ŹŹ*
transformer_block_1_3834020:	Ź/
transformer_block_1_3834022:
ŹŹ*
transformer_block_1_3834024:	ŹF
3sequential_2_dense_4_matmul_readvariableop_resource:	ŹB
4sequential_2_dense_4_biasadd_readvariableop_resource:
identity˘+sequential_2/dense_4/BiasAdd/ReadVariableOp˘*sequential_2/dense_4/MatMul/ReadVariableOp˘7token_and_position_embedding/embedding/embedding_lookup˘9token_and_position_embedding/embedding_1/embedding_lookup˘)transformer_block/StatefulPartitionedCall˘+transformer_block_1/StatefulPartitionedCallj
(token_and_position_embedding/range/startConst*
_output_shapes
: *
dtype0*
value	B : j
(token_and_position_embedding/range/limitConst*
_output_shapes
: *
dtype0*
value	B :dj
(token_and_position_embedding/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :ŕ
"token_and_position_embedding/rangeRange1token_and_position_embedding/range/start:output:01token_and_position_embedding/range/limit:output:01token_and_position_embedding/range/delta:output:0*
_output_shapes
:dĎ
9token_and_position_embedding/embedding_1/embedding_lookupResourceGatherAtoken_and_position_embedding_embedding_1_embedding_lookup_3833964+token_and_position_embedding/range:output:0*
Tindices0*T
_classJ
HFloc:@token_and_position_embedding/embedding_1/embedding_lookup/3833964*
_output_shapes
:	dŹ*
dtype0
Btoken_and_position_embedding/embedding_1/embedding_lookup/IdentityIdentityBtoken_and_position_embedding/embedding_1/embedding_lookup:output:0*
T0*T
_classJ
HFloc:@token_and_position_embedding/embedding_1/embedding_lookup/3833964*
_output_shapes
:	dŹÇ
Dtoken_and_position_embedding/embedding_1/embedding_lookup/Identity_1IdentityKtoken_and_position_embedding/embedding_1/embedding_lookup/Identity:output:0*
T0*
_output_shapes
:	dŹŤ
7token_and_position_embedding/embedding/embedding_lookupResourceGather?token_and_position_embedding_embedding_embedding_lookup_3833969x_in*
Tindices0*R
_classH
FDloc:@token_and_position_embedding/embedding/embedding_lookup/3833969*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ź*
dtype0
@token_and_position_embedding/embedding/embedding_lookup/IdentityIdentity@token_and_position_embedding/embedding/embedding_lookup:output:0*
T0*R
_classH
FDloc:@token_and_position_embedding/embedding/embedding_lookup/3833969*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ŹĚ
Btoken_and_position_embedding/embedding/embedding_lookup/Identity_1IdentityItoken_and_position_embedding/embedding/embedding_lookup/Identity:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Źď
 token_and_position_embedding/addAddV2Ktoken_and_position_embedding/embedding/embedding_lookup/Identity_1:output:0Mtoken_and_position_embedding/embedding_1/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:	dŹĆ
)transformer_block/StatefulPartitionedCallStatefulPartitionedCall$token_and_position_embedding/add:z:0transformer_block_3833975transformer_block_3833977transformer_block_3833979transformer_block_3833981transformer_block_3833983transformer_block_3833985transformer_block_3833987transformer_block_3833989transformer_block_3833991*
Tin
2
*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:ddŹ*+
_read_only_resource_inputs
		*-
config_proto

CPU

GPU 2J 8 *!
fR
__inference_call_3827515Z
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @
dropout/dropout/MulMul2transformer_block/StatefulPartitionedCall:output:0dropout/dropout/Const:output:0*
T0*#
_output_shapes
:ddŹj
dropout/dropout/ShapeConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ,  
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*#
_output_shapes
:ddŹ*
dtype0c
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?ş
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*#
_output_shapes
:ddŹ\
dropout/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ż
dropout/dropout/SelectV2SelectV2 dropout/dropout/GreaterEqual:z:0dropout/dropout/Mul:z:0 dropout/dropout/Const_1:output:0*
T0*#
_output_shapes
:ddŹŰ
+transformer_block_1/StatefulPartitionedCallStatefulPartitionedCall!dropout/dropout/SelectV2:output:0$token_and_position_embedding/add:z:0transformer_block_1_3834002transformer_block_1_3834004transformer_block_1_3834006transformer_block_1_3834008transformer_block_1_3834010transformer_block_1_3834012transformer_block_1_3834014transformer_block_1_3834016transformer_block_1_3834018transformer_block_1_3834020transformer_block_1_3834022transformer_block_1_3834024*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:ddŹ*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *!
fR
__inference_call_3827796P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 

ExpandDims
ExpandDims4transformer_block_1/StatefulPartitionedCall:output:0ExpandDims/dim:output:0*
T0*'
_output_shapes
:ddŹ{
*global_max_pooling2d/Max/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      
global_max_pooling2d/MaxMaxExpandDims:output:03global_max_pooling2d/Max/reduction_indices:output:0*
T0*
_output_shapes
:	Ź
*sequential_2/dense_4/MatMul/ReadVariableOpReadVariableOp3sequential_2_dense_4_matmul_readvariableop_resource*
_output_shapes
:	Ź*
dtype0Ľ
sequential_2/dense_4/MatMulMatMul!global_max_pooling2d/Max:output:02sequential_2/dense_4/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:
+sequential_2/dense_4/BiasAdd/ReadVariableOpReadVariableOp4sequential_2_dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ź
sequential_2/dense_4/BiasAddBiasAdd%sequential_2/dense_4/MatMul:product:03sequential_2/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:w
sequential_2/dense_4/SigmoidSigmoid%sequential_2/dense_4/BiasAdd:output:0*
T0*
_output_shapes

:f
IdentityIdentity sequential_2/dense_4/Sigmoid:y:0^NoOp*
T0*
_output_shapes

:ń
NoOpNoOp,^sequential_2/dense_4/BiasAdd/ReadVariableOp+^sequential_2/dense_4/MatMul/ReadVariableOp8^token_and_position_embedding/embedding/embedding_lookup:^token_and_position_embedding/embedding_1/embedding_lookup*^transformer_block/StatefulPartitionedCall,^transformer_block_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:˙˙˙˙˙˙˙˙˙: : : : : : : : : : : : : : : : : : : : : : : : : 2Z
+sequential_2/dense_4/BiasAdd/ReadVariableOp+sequential_2/dense_4/BiasAdd/ReadVariableOp2X
*sequential_2/dense_4/MatMul/ReadVariableOp*sequential_2/dense_4/MatMul/ReadVariableOp2r
7token_and_position_embedding/embedding/embedding_lookup7token_and_position_embedding/embedding/embedding_lookup2v
9token_and_position_embedding/embedding_1/embedding_lookup9token_and_position_embedding/embedding_1/embedding_lookup2V
)transformer_block/StatefulPartitionedCall)transformer_block/StatefulPartitionedCall2Z
+transformer_block_1/StatefulPartitionedCall+transformer_block_1/StatefulPartitionedCall:I E
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_namex_in
Ľ;
ů
 __inference__traced_save_3835404
file_prefixh
dsavev2_emotion_detection_model_token_and_position_embedding_embedding_embeddings_read_readvariableopj
fsavev2_emotion_detection_model_token_and_position_embedding_embedding_1_embeddings_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop*
&savev2_variable_11_read_readvariableop*
&savev2_variable_10_read_readvariableop)
%savev2_variable_9_read_readvariableop)
%savev2_variable_8_read_readvariableop)
%savev2_variable_7_read_readvariableop)
%savev2_variable_6_read_readvariableop8
4savev2_layer_normalization_gamma_read_readvariableop7
3savev2_layer_normalization_beta_read_readvariableop-
)savev2_dense_2_kernel_read_readvariableop+
'savev2_dense_2_bias_read_readvariableop-
)savev2_dense_3_kernel_read_readvariableop+
'savev2_dense_3_bias_read_readvariableop)
%savev2_variable_5_read_readvariableop)
%savev2_variable_4_read_readvariableop)
%savev2_variable_3_read_readvariableop)
%savev2_variable_2_read_readvariableop)
%savev2_variable_1_read_readvariableop'
#savev2_variable_read_readvariableop:
6savev2_layer_normalization_1_gamma_read_readvariableop9
5savev2_layer_normalization_1_beta_read_readvariableop-
)savev2_dense_4_kernel_read_readvariableop+
'savev2_dense_4_bias_read_readvariableop
savev2_const

identity_1˘MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ü	
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Ľ	
value	B	B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH§
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*M
valueDBBB B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0dsavev2_emotion_detection_model_token_and_position_embedding_embedding_embeddings_read_readvariableopfsavev2_emotion_detection_model_token_and_position_embedding_embedding_1_embeddings_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop&savev2_variable_11_read_readvariableop&savev2_variable_10_read_readvariableop%savev2_variable_9_read_readvariableop%savev2_variable_8_read_readvariableop%savev2_variable_7_read_readvariableop%savev2_variable_6_read_readvariableop4savev2_layer_normalization_gamma_read_readvariableop3savev2_layer_normalization_beta_read_readvariableop)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableop)savev2_dense_3_kernel_read_readvariableop'savev2_dense_3_bias_read_readvariableop%savev2_variable_5_read_readvariableop%savev2_variable_4_read_readvariableop%savev2_variable_3_read_readvariableop%savev2_variable_2_read_readvariableop%savev2_variable_1_read_readvariableop#savev2_variable_read_readvariableop6savev2_layer_normalization_1_gamma_read_readvariableop5savev2_layer_normalization_1_beta_read_readvariableop)savev2_dense_4_kernel_read_readvariableop'savev2_dense_4_bias_read_readvariableopsavev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *+
dtypes!
2
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:ł
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*š
_input_shapes§
¤: :
XŹ:	dŹ:
ŹŹ:Ź:
ŹŹ:Ź:
ŹŹ:
ŹŹ:
ŹŹ:
ŹŹ:
ŹŹ:
ŹŹ:Ź:Ź:
ŹŹ:Ź:
ŹŹ:Ź:
ŹŹ:
ŹŹ:
ŹŹ:
ŹŹ:
ŹŹ:
ŹŹ:Ź:Ź:	Ź:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:&"
 
_output_shapes
:
XŹ:%!

_output_shapes
:	dŹ:&"
 
_output_shapes
:
ŹŹ:!

_output_shapes	
:Ź:&"
 
_output_shapes
:
ŹŹ:!

_output_shapes	
:Ź:&"
 
_output_shapes
:
ŹŹ:&"
 
_output_shapes
:
ŹŹ:&	"
 
_output_shapes
:
ŹŹ:&
"
 
_output_shapes
:
ŹŹ:&"
 
_output_shapes
:
ŹŹ:&"
 
_output_shapes
:
ŹŹ:!

_output_shapes	
:Ź:!

_output_shapes	
:Ź:&"
 
_output_shapes
:
ŹŹ:!

_output_shapes	
:Ź:&"
 
_output_shapes
:
ŹŹ:!

_output_shapes	
:Ź:&"
 
_output_shapes
:
ŹŹ:&"
 
_output_shapes
:
ŹŹ:&"
 
_output_shapes
:
ŹŹ:&"
 
_output_shapes
:
ŹŹ:&"
 
_output_shapes
:
ŹŹ:&"
 
_output_shapes
:
ŹŹ:!

_output_shapes	
:Ź:!

_output_shapes	
:Ź:%!

_output_shapes
:	Ź: 

_output_shapes
::

_output_shapes
: 
Ď

.__inference_sequential_2_layer_call_fn_3834790

inputs
unknown:	Ź
	unknown_0:
identity˘StatefulPartitionedCallŢ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sequential_2_layer_call_and_return_conditional_losses_3832454o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:˙˙˙˙˙˙˙˙˙Ź: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ź
 
_user_specified_nameinputs
Ů
ţ
D__inference_dense_1_layer_call_and_return_conditional_losses_3835218

inputs5
!tensordot_readvariableop_resource:
ŹŹ.
biasadd_readvariableop_resource:	Ź
identity˘BiasAdd/ReadVariableOp˘Tensordot/ReadVariableOp|
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource* 
_output_shapes
:
ŹŹ*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ť
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ż
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:z
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙dŹ
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ź\
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:ŹY
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : §
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙dŹs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:Ź*
dtype0}
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙dŹd
IdentityIdentityBiasAdd:output:0^NoOp*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙dŹz
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:˙˙˙˙˙˙˙˙˙dŹ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:˙˙˙˙˙˙˙˙˙dŹ
 
_user_specified_nameinputs
łŔ
ç

P__inference_transformer_block_1_layer_call_and_return_conditional_losses_3833131

inputs
context_sequence,
attention_head_2_3832967:
ŹŹ,
attention_head_2_3832969:
ŹŹ,
attention_head_2_3832971:
ŹŹA
2layer_normalization_1_cast_readvariableop_resource:	ŹC
4layer_normalization_1_cast_1_readvariableop_resource:	Ź,
attention_head_3_3833018:
ŹŹ,
attention_head_3_3833020:
ŹŹ,
attention_head_3_3833022:
ŹŹJ
6sequential_1_dense_2_tensordot_readvariableop_resource:
ŹŹC
4sequential_1_dense_2_biasadd_readvariableop_resource:	ŹJ
6sequential_1_dense_3_tensordot_readvariableop_resource:
ŹŹC
4sequential_1_dense_3_biasadd_readvariableop_resource:	Ź
identity˘(attention_head_2/StatefulPartitionedCall˘(attention_head_3/StatefulPartitionedCall˘)layer_normalization_1/Cast/ReadVariableOp˘+layer_normalization_1/Cast_1/ReadVariableOp˘+layer_normalization_1/Cast_2/ReadVariableOp˘+layer_normalization_1/Cast_3/ReadVariableOp˘+layer_normalization_1/Cast_4/ReadVariableOp˘+layer_normalization_1/Cast_5/ReadVariableOp˘+sequential_1/dense_2/BiasAdd/ReadVariableOp˘-sequential_1/dense_2/Tensordot/ReadVariableOp˘+sequential_1/dense_3/BiasAdd/ReadVariableOp˘-sequential_1/dense_3/Tensordot/ReadVariableOp
(attention_head_2/StatefulPartitionedCallStatefulPartitionedCallinputsinputsinputsattention_head_2_3832967attention_head_2_3832969attention_head_2_3832971*
Tin

2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:ddŹ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *!
fR
__inference_call_3827587u
addAddV21attention_head_2/StatefulPartitionedCall:output:0inputs*
T0*#
_output_shapes
:ddŹp
layer_normalization_1/ShapeConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ,  s
)layer_normalization_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+layer_normalization_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+layer_normalization_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ż
#layer_normalization_1/strided_sliceStridedSlice$layer_normalization_1/Shape:output:02layer_normalization_1/strided_slice/stack:output:04layer_normalization_1/strided_slice/stack_1:output:04layer_normalization_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
layer_normalization_1/mul/xConst*
_output_shapes
: *
dtype0*
value	B :
layer_normalization_1/mulMul$layer_normalization_1/mul/x:output:0,layer_normalization_1/strided_slice:output:0*
T0*
_output_shapes
: u
+layer_normalization_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ç
%layer_normalization_1/strided_slice_1StridedSlice$layer_normalization_1/Shape:output:04layer_normalization_1/strided_slice_1/stack:output:06layer_normalization_1/strided_slice_1/stack_1:output:06layer_normalization_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
layer_normalization_1/mul_1Mullayer_normalization_1/mul:z:0.layer_normalization_1/strided_slice_1:output:0*
T0*
_output_shapes
: u
+layer_normalization_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ç
%layer_normalization_1/strided_slice_2StridedSlice$layer_normalization_1/Shape:output:04layer_normalization_1/strided_slice_2/stack:output:06layer_normalization_1/strided_slice_2/stack_1:output:06layer_normalization_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
layer_normalization_1/mul_2/xConst*
_output_shapes
: *
dtype0*
value	B :
layer_normalization_1/mul_2Mul&layer_normalization_1/mul_2/x:output:0.layer_normalization_1/strided_slice_2:output:0*
T0*
_output_shapes
: g
%layer_normalization_1/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :g
%layer_normalization_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :ű
#layer_normalization_1/Reshape/shapePack.layer_normalization_1/Reshape/shape/0:output:0layer_normalization_1/mul_1:z:0layer_normalization_1/mul_2:z:0.layer_normalization_1/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:
layer_normalization_1/ReshapeReshapeadd:z:0,layer_normalization_1/Reshape/shape:output:0*
T0*(
_output_shapes
:NŹx
!layer_normalization_1/ones/packedPacklayer_normalization_1/mul_1:z:0*
N*
T0*
_output_shapes
:e
 layer_normalization_1/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
layer_normalization_1/onesFill*layer_normalization_1/ones/packed:output:0)layer_normalization_1/ones/Const:output:0*
T0*
_output_shapes	
:Ny
"layer_normalization_1/zeros/packedPacklayer_normalization_1/mul_1:z:0*
N*
T0*
_output_shapes
:f
!layer_normalization_1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ˘
layer_normalization_1/zerosFill+layer_normalization_1/zeros/packed:output:0*layer_normalization_1/zeros/Const:output:0*
T0*
_output_shapes	
:N^
layer_normalization_1/ConstConst*
_output_shapes
: *
dtype0*
valueB `
layer_normalization_1/Const_1Const*
_output_shapes
: *
dtype0*
valueB 
&layer_normalization_1/FusedBatchNormV3FusedBatchNormV3&layer_normalization_1/Reshape:output:0#layer_normalization_1/ones:output:0$layer_normalization_1/zeros:output:0$layer_normalization_1/Const:output:0&layer_normalization_1/Const_1:output:0*
T0*
U0*H
_output_shapes6
4:NŹ:N:N:N:N:*
data_formatNCHW*
epsilon%o:Ş
layer_normalization_1/Reshape_1Reshape*layer_normalization_1/FusedBatchNormV3:y:0$layer_normalization_1/Shape:output:0*
T0*#
_output_shapes
:ddŹ
)layer_normalization_1/Cast/ReadVariableOpReadVariableOp2layer_normalization_1_cast_readvariableop_resource*
_output_shapes	
:Ź*
dtype0­
layer_normalization_1/mul_3Mul(layer_normalization_1/Reshape_1:output:01layer_normalization_1/Cast/ReadVariableOp:value:0*
T0*#
_output_shapes
:ddŹ
+layer_normalization_1/Cast_1/ReadVariableOpReadVariableOp4layer_normalization_1_cast_1_readvariableop_resource*
_output_shapes	
:Ź*
dtype0Ś
layer_normalization_1/addAddV2layer_normalization_1/mul_3:z:03layer_normalization_1/Cast_1/ReadVariableOp:value:0*
T0*#
_output_shapes
:ddŹŔ
(attention_head_3/StatefulPartitionedCallStatefulPartitionedCalllayer_normalization_1/add:z:0layer_normalization_1/add:z:0context_sequenceattention_head_3_3833018attention_head_3_3833020attention_head_3_3833022*
Tin

2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:ddŹ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *!
fR
__inference_call_3827682
add_1AddV21attention_head_3/StatefulPartitionedCall:output:0layer_normalization_1/add:z:0*
T0*#
_output_shapes
:ddŹr
layer_normalization_1/Shape_1Const*
_output_shapes
:*
dtype0*!
valueB"d   d   ,  u
+layer_normalization_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-layer_normalization_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:É
%layer_normalization_1/strided_slice_3StridedSlice&layer_normalization_1/Shape_1:output:04layer_normalization_1/strided_slice_3/stack:output:06layer_normalization_1/strided_slice_3/stack_1:output:06layer_normalization_1/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
layer_normalization_1/mul_4/xConst*
_output_shapes
: *
dtype0*
value	B :
layer_normalization_1/mul_4Mul&layer_normalization_1/mul_4/x:output:0.layer_normalization_1/strided_slice_3:output:0*
T0*
_output_shapes
: u
+layer_normalization_1/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_1/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_1/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:É
%layer_normalization_1/strided_slice_4StridedSlice&layer_normalization_1/Shape_1:output:04layer_normalization_1/strided_slice_4/stack:output:06layer_normalization_1/strided_slice_4/stack_1:output:06layer_normalization_1/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
layer_normalization_1/mul_5Mullayer_normalization_1/mul_4:z:0.layer_normalization_1/strided_slice_4:output:0*
T0*
_output_shapes
: u
+layer_normalization_1/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_1/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_1/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:É
%layer_normalization_1/strided_slice_5StridedSlice&layer_normalization_1/Shape_1:output:04layer_normalization_1/strided_slice_5/stack:output:06layer_normalization_1/strided_slice_5/stack_1:output:06layer_normalization_1/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
layer_normalization_1/mul_6/xConst*
_output_shapes
: *
dtype0*
value	B :
layer_normalization_1/mul_6Mul&layer_normalization_1/mul_6/x:output:0.layer_normalization_1/strided_slice_5:output:0*
T0*
_output_shapes
: i
'layer_normalization_1/Reshape_2/shape/0Const*
_output_shapes
: *
dtype0*
value	B :i
'layer_normalization_1/Reshape_2/shape/3Const*
_output_shapes
: *
dtype0*
value	B :
%layer_normalization_1/Reshape_2/shapePack0layer_normalization_1/Reshape_2/shape/0:output:0layer_normalization_1/mul_5:z:0layer_normalization_1/mul_6:z:00layer_normalization_1/Reshape_2/shape/3:output:0*
N*
T0*
_output_shapes
:
layer_normalization_1/Reshape_2Reshape	add_1:z:0.layer_normalization_1/Reshape_2/shape:output:0*
T0*(
_output_shapes
:NŹz
#layer_normalization_1/ones_1/packedPacklayer_normalization_1/mul_5:z:0*
N*
T0*
_output_shapes
:g
"layer_normalization_1/ones_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ľ
layer_normalization_1/ones_1Fill,layer_normalization_1/ones_1/packed:output:0+layer_normalization_1/ones_1/Const:output:0*
T0*
_output_shapes	
:N{
$layer_normalization_1/zeros_1/packedPacklayer_normalization_1/mul_5:z:0*
N*
T0*
_output_shapes
:h
#layer_normalization_1/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ¨
layer_normalization_1/zeros_1Fill-layer_normalization_1/zeros_1/packed:output:0,layer_normalization_1/zeros_1/Const:output:0*
T0*
_output_shapes	
:N`
layer_normalization_1/Const_2Const*
_output_shapes
: *
dtype0*
valueB `
layer_normalization_1/Const_3Const*
_output_shapes
: *
dtype0*
valueB 
(layer_normalization_1/FusedBatchNormV3_1FusedBatchNormV3(layer_normalization_1/Reshape_2:output:0%layer_normalization_1/ones_1:output:0&layer_normalization_1/zeros_1:output:0&layer_normalization_1/Const_2:output:0&layer_normalization_1/Const_3:output:0*
T0*
U0*H
_output_shapes6
4:NŹ:N:N:N:N:*
data_formatNCHW*
epsilon%o:Ž
layer_normalization_1/Reshape_3Reshape,layer_normalization_1/FusedBatchNormV3_1:y:0&layer_normalization_1/Shape_1:output:0*
T0*#
_output_shapes
:ddŹ
+layer_normalization_1/Cast_2/ReadVariableOpReadVariableOp2layer_normalization_1_cast_readvariableop_resource*
_output_shapes	
:Ź*
dtype0Ż
layer_normalization_1/mul_7Mul(layer_normalization_1/Reshape_3:output:03layer_normalization_1/Cast_2/ReadVariableOp:value:0*
T0*#
_output_shapes
:ddŹ
+layer_normalization_1/Cast_3/ReadVariableOpReadVariableOp4layer_normalization_1_cast_1_readvariableop_resource*
_output_shapes	
:Ź*
dtype0¨
layer_normalization_1/add_1AddV2layer_normalization_1/mul_7:z:03layer_normalization_1/Cast_3/ReadVariableOp:value:0*
T0*#
_output_shapes
:ddŹŚ
-sequential_1/dense_2/Tensordot/ReadVariableOpReadVariableOp6sequential_1_dense_2_tensordot_readvariableop_resource* 
_output_shapes
:
ŹŹ*
dtype0}
,sequential_1/dense_2/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"'  ,  ´
&sequential_1/dense_2/Tensordot/ReshapeReshapelayer_normalization_1/add_1:z:05sequential_1/dense_2/Tensordot/Reshape/shape:output:0*
T0* 
_output_shapes
:
NŹÂ
%sequential_1/dense_2/Tensordot/MatMulMatMul/sequential_1/dense_2/Tensordot/Reshape:output:05sequential_1/dense_2/Tensordot/ReadVariableOp:value:0*
T0* 
_output_shapes
:
NŹy
$sequential_1/dense_2/Tensordot/shapeConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ,  ˇ
sequential_1/dense_2/TensordotReshape/sequential_1/dense_2/Tensordot/MatMul:product:0-sequential_1/dense_2/Tensordot/shape:output:0*
T0*#
_output_shapes
:ddŹ
+sequential_1/dense_2/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:Ź*
dtype0ł
sequential_1/dense_2/BiasAddBiasAdd'sequential_1/dense_2/Tensordot:output:03sequential_1/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*#
_output_shapes
:ddŹw
sequential_1/dense_2/LeakyRelu	LeakyRelu%sequential_1/dense_2/BiasAdd:output:0*#
_output_shapes
:ddŹŚ
-sequential_1/dense_3/Tensordot/ReadVariableOpReadVariableOp6sequential_1_dense_3_tensordot_readvariableop_resource* 
_output_shapes
:
ŹŹ*
dtype0}
,sequential_1/dense_3/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"'  ,  Á
&sequential_1/dense_3/Tensordot/ReshapeReshape,sequential_1/dense_2/LeakyRelu:activations:05sequential_1/dense_3/Tensordot/Reshape/shape:output:0*
T0* 
_output_shapes
:
NŹÂ
%sequential_1/dense_3/Tensordot/MatMulMatMul/sequential_1/dense_3/Tensordot/Reshape:output:05sequential_1/dense_3/Tensordot/ReadVariableOp:value:0*
T0* 
_output_shapes
:
NŹy
$sequential_1/dense_3/Tensordot/shapeConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ,  ˇ
sequential_1/dense_3/TensordotReshape/sequential_1/dense_3/Tensordot/MatMul:product:0-sequential_1/dense_3/Tensordot/shape:output:0*
T0*#
_output_shapes
:ddŹ
+sequential_1/dense_3/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_3_biasadd_readvariableop_resource*
_output_shapes	
:Ź*
dtype0ł
sequential_1/dense_3/BiasAddBiasAdd'sequential_1/dense_3/Tensordot:output:03sequential_1/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*#
_output_shapes
:ddŹ
add_2AddV2%sequential_1/dense_3/BiasAdd:output:0layer_normalization_1/add_1:z:0*
T0*#
_output_shapes
:ddŹr
layer_normalization_1/Shape_2Const*
_output_shapes
:*
dtype0*!
valueB"d   d   ,  u
+layer_normalization_1/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-layer_normalization_1/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_1/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB:É
%layer_normalization_1/strided_slice_6StridedSlice&layer_normalization_1/Shape_2:output:04layer_normalization_1/strided_slice_6/stack:output:06layer_normalization_1/strided_slice_6/stack_1:output:06layer_normalization_1/strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
layer_normalization_1/mul_8/xConst*
_output_shapes
: *
dtype0*
value	B :
layer_normalization_1/mul_8Mul&layer_normalization_1/mul_8/x:output:0.layer_normalization_1/strided_slice_6:output:0*
T0*
_output_shapes
: u
+layer_normalization_1/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_1/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_1/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB:É
%layer_normalization_1/strided_slice_7StridedSlice&layer_normalization_1/Shape_2:output:04layer_normalization_1/strided_slice_7/stack:output:06layer_normalization_1/strided_slice_7/stack_1:output:06layer_normalization_1/strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
layer_normalization_1/mul_9Mullayer_normalization_1/mul_8:z:0.layer_normalization_1/strided_slice_7:output:0*
T0*
_output_shapes
: u
+layer_normalization_1/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_1/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_1/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:É
%layer_normalization_1/strided_slice_8StridedSlice&layer_normalization_1/Shape_2:output:04layer_normalization_1/strided_slice_8/stack:output:06layer_normalization_1/strided_slice_8/stack_1:output:06layer_normalization_1/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask`
layer_normalization_1/mul_10/xConst*
_output_shapes
: *
dtype0*
value	B :
layer_normalization_1/mul_10Mul'layer_normalization_1/mul_10/x:output:0.layer_normalization_1/strided_slice_8:output:0*
T0*
_output_shapes
: i
'layer_normalization_1/Reshape_4/shape/0Const*
_output_shapes
: *
dtype0*
value	B :i
'layer_normalization_1/Reshape_4/shape/3Const*
_output_shapes
: *
dtype0*
value	B :
%layer_normalization_1/Reshape_4/shapePack0layer_normalization_1/Reshape_4/shape/0:output:0layer_normalization_1/mul_9:z:0 layer_normalization_1/mul_10:z:00layer_normalization_1/Reshape_4/shape/3:output:0*
N*
T0*
_output_shapes
:
layer_normalization_1/Reshape_4Reshape	add_2:z:0.layer_normalization_1/Reshape_4/shape:output:0*
T0*(
_output_shapes
:NŹz
#layer_normalization_1/ones_2/packedPacklayer_normalization_1/mul_9:z:0*
N*
T0*
_output_shapes
:g
"layer_normalization_1/ones_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ľ
layer_normalization_1/ones_2Fill,layer_normalization_1/ones_2/packed:output:0+layer_normalization_1/ones_2/Const:output:0*
T0*
_output_shapes	
:N{
$layer_normalization_1/zeros_2/packedPacklayer_normalization_1/mul_9:z:0*
N*
T0*
_output_shapes
:h
#layer_normalization_1/zeros_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ¨
layer_normalization_1/zeros_2Fill-layer_normalization_1/zeros_2/packed:output:0,layer_normalization_1/zeros_2/Const:output:0*
T0*
_output_shapes	
:N`
layer_normalization_1/Const_4Const*
_output_shapes
: *
dtype0*
valueB `
layer_normalization_1/Const_5Const*
_output_shapes
: *
dtype0*
valueB 
(layer_normalization_1/FusedBatchNormV3_2FusedBatchNormV3(layer_normalization_1/Reshape_4:output:0%layer_normalization_1/ones_2:output:0&layer_normalization_1/zeros_2:output:0&layer_normalization_1/Const_4:output:0&layer_normalization_1/Const_5:output:0*
T0*
U0*H
_output_shapes6
4:NŹ:N:N:N:N:*
data_formatNCHW*
epsilon%o:Ž
layer_normalization_1/Reshape_5Reshape,layer_normalization_1/FusedBatchNormV3_2:y:0&layer_normalization_1/Shape_2:output:0*
T0*#
_output_shapes
:ddŹ
+layer_normalization_1/Cast_4/ReadVariableOpReadVariableOp2layer_normalization_1_cast_readvariableop_resource*
_output_shapes	
:Ź*
dtype0°
layer_normalization_1/mul_11Mul(layer_normalization_1/Reshape_5:output:03layer_normalization_1/Cast_4/ReadVariableOp:value:0*
T0*#
_output_shapes
:ddŹ
+layer_normalization_1/Cast_5/ReadVariableOpReadVariableOp4layer_normalization_1_cast_1_readvariableop_resource*
_output_shapes	
:Ź*
dtype0Š
layer_normalization_1/add_2AddV2 layer_normalization_1/mul_11:z:03layer_normalization_1/Cast_5/ReadVariableOp:value:0*
T0*#
_output_shapes
:ddŹj
IdentityIdentitylayer_normalization_1/add_2:z:0^NoOp*
T0*#
_output_shapes
:ddŹę
NoOpNoOp)^attention_head_2/StatefulPartitionedCall)^attention_head_3/StatefulPartitionedCall*^layer_normalization_1/Cast/ReadVariableOp,^layer_normalization_1/Cast_1/ReadVariableOp,^layer_normalization_1/Cast_2/ReadVariableOp,^layer_normalization_1/Cast_3/ReadVariableOp,^layer_normalization_1/Cast_4/ReadVariableOp,^layer_normalization_1/Cast_5/ReadVariableOp,^sequential_1/dense_2/BiasAdd/ReadVariableOp.^sequential_1/dense_2/Tensordot/ReadVariableOp,^sequential_1/dense_3/BiasAdd/ReadVariableOp.^sequential_1/dense_3/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:ddŹ:	dŹ: : : : : : : : : : : : 2T
(attention_head_2/StatefulPartitionedCall(attention_head_2/StatefulPartitionedCall2T
(attention_head_3/StatefulPartitionedCall(attention_head_3/StatefulPartitionedCall2V
)layer_normalization_1/Cast/ReadVariableOp)layer_normalization_1/Cast/ReadVariableOp2Z
+layer_normalization_1/Cast_1/ReadVariableOp+layer_normalization_1/Cast_1/ReadVariableOp2Z
+layer_normalization_1/Cast_2/ReadVariableOp+layer_normalization_1/Cast_2/ReadVariableOp2Z
+layer_normalization_1/Cast_3/ReadVariableOp+layer_normalization_1/Cast_3/ReadVariableOp2Z
+layer_normalization_1/Cast_4/ReadVariableOp+layer_normalization_1/Cast_4/ReadVariableOp2Z
+layer_normalization_1/Cast_5/ReadVariableOp+layer_normalization_1/Cast_5/ReadVariableOp2Z
+sequential_1/dense_2/BiasAdd/ReadVariableOp+sequential_1/dense_2/BiasAdd/ReadVariableOp2^
-sequential_1/dense_2/Tensordot/ReadVariableOp-sequential_1/dense_2/Tensordot/ReadVariableOp2Z
+sequential_1/dense_3/BiasAdd/ReadVariableOp+sequential_1/dense_3/BiasAdd/ReadVariableOp2^
-sequential_1/dense_3/Tensordot/ReadVariableOp-sequential_1/dense_3/Tensordot/ReadVariableOp:K G
#
_output_shapes
:ddŹ
 
_user_specified_nameinputs:QM

_output_shapes
:	dŹ
*
_user_specified_namecontext_sequence
§

N__inference_transformer_block_layer_call_and_return_conditional_losses_3834365

inputs*
attention_head_3834250:
ŹŹ*
attention_head_3834252:
ŹŹ*
attention_head_3834254:
ŹŹ?
0layer_normalization_cast_readvariableop_resource:	ŹA
2layer_normalization_cast_1_readvariableop_resource:	ŹF
2sequential_dense_tensordot_readvariableop_resource:
ŹŹ?
0sequential_dense_biasadd_readvariableop_resource:	ŹH
4sequential_dense_1_tensordot_readvariableop_resource:
ŹŹA
2sequential_dense_1_biasadd_readvariableop_resource:	Ź
identity˘&attention_head/StatefulPartitionedCall˘'layer_normalization/Cast/ReadVariableOp˘)layer_normalization/Cast_1/ReadVariableOp˘)layer_normalization/Cast_2/ReadVariableOp˘)layer_normalization/Cast_3/ReadVariableOp˘'sequential/dense/BiasAdd/ReadVariableOp˘)sequential/dense/Tensordot/ReadVariableOp˘)sequential/dense_1/BiasAdd/ReadVariableOp˘+sequential/dense_1/Tensordot/ReadVariableOp
&attention_head/StatefulPartitionedCallStatefulPartitionedCallinputsinputsinputsattention_head_3834250attention_head_3834252attention_head_3834254*
Tin

2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:ddŹ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *!
fR
__inference_call_3827399s
addAddV2/attention_head/StatefulPartitionedCall:output:0inputs*
T0*#
_output_shapes
:ddŹn
layer_normalization/ShapeConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ,  q
'layer_normalization/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)layer_normalization/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)layer_normalization/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ľ
!layer_normalization/strided_sliceStridedSlice"layer_normalization/Shape:output:00layer_normalization/strided_slice/stack:output:02layer_normalization/strided_slice/stack_1:output:02layer_normalization/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask[
layer_normalization/mul/xConst*
_output_shapes
: *
dtype0*
value	B :
layer_normalization/mulMul"layer_normalization/mul/x:output:0*layer_normalization/strided_slice:output:0*
T0*
_output_shapes
: s
)layer_normalization/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:u
+layer_normalization/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+layer_normalization/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:˝
#layer_normalization/strided_slice_1StridedSlice"layer_normalization/Shape:output:02layer_normalization/strided_slice_1/stack:output:04layer_normalization/strided_slice_1/stack_1:output:04layer_normalization/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
layer_normalization/mul_1Mullayer_normalization/mul:z:0,layer_normalization/strided_slice_1:output:0*
T0*
_output_shapes
: s
)layer_normalization/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:u
+layer_normalization/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+layer_normalization/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:˝
#layer_normalization/strided_slice_2StridedSlice"layer_normalization/Shape:output:02layer_normalization/strided_slice_2/stack:output:04layer_normalization/strided_slice_2/stack_1:output:04layer_normalization/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
layer_normalization/mul_2/xConst*
_output_shapes
: *
dtype0*
value	B :
layer_normalization/mul_2Mul$layer_normalization/mul_2/x:output:0,layer_normalization/strided_slice_2:output:0*
T0*
_output_shapes
: e
#layer_normalization/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :e
#layer_normalization/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :ń
!layer_normalization/Reshape/shapePack,layer_normalization/Reshape/shape/0:output:0layer_normalization/mul_1:z:0layer_normalization/mul_2:z:0,layer_normalization/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:
layer_normalization/ReshapeReshapeadd:z:0*layer_normalization/Reshape/shape:output:0*
T0*(
_output_shapes
:NŹt
layer_normalization/ones/packedPacklayer_normalization/mul_1:z:0*
N*
T0*
_output_shapes
:c
layer_normalization/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
layer_normalization/onesFill(layer_normalization/ones/packed:output:0'layer_normalization/ones/Const:output:0*
T0*
_output_shapes	
:Nu
 layer_normalization/zeros/packedPacklayer_normalization/mul_1:z:0*
N*
T0*
_output_shapes
:d
layer_normalization/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
layer_normalization/zerosFill)layer_normalization/zeros/packed:output:0(layer_normalization/zeros/Const:output:0*
T0*
_output_shapes	
:N\
layer_normalization/ConstConst*
_output_shapes
: *
dtype0*
valueB ^
layer_normalization/Const_1Const*
_output_shapes
: *
dtype0*
valueB ô
$layer_normalization/FusedBatchNormV3FusedBatchNormV3$layer_normalization/Reshape:output:0!layer_normalization/ones:output:0"layer_normalization/zeros:output:0"layer_normalization/Const:output:0$layer_normalization/Const_1:output:0*
T0*
U0*H
_output_shapes6
4:NŹ:N:N:N:N:*
data_formatNCHW*
epsilon%o:¤
layer_normalization/Reshape_1Reshape(layer_normalization/FusedBatchNormV3:y:0"layer_normalization/Shape:output:0*
T0*#
_output_shapes
:ddŹ
'layer_normalization/Cast/ReadVariableOpReadVariableOp0layer_normalization_cast_readvariableop_resource*
_output_shapes	
:Ź*
dtype0§
layer_normalization/mul_3Mul&layer_normalization/Reshape_1:output:0/layer_normalization/Cast/ReadVariableOp:value:0*
T0*#
_output_shapes
:ddŹ
)layer_normalization/Cast_1/ReadVariableOpReadVariableOp2layer_normalization_cast_1_readvariableop_resource*
_output_shapes	
:Ź*
dtype0 
layer_normalization/addAddV2layer_normalization/mul_3:z:01layer_normalization/Cast_1/ReadVariableOp:value:0*
T0*#
_output_shapes
:ddŹ
)sequential/dense/Tensordot/ReadVariableOpReadVariableOp2sequential_dense_tensordot_readvariableop_resource* 
_output_shapes
:
ŹŹ*
dtype0y
(sequential/dense/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"'  ,  ¨
"sequential/dense/Tensordot/ReshapeReshapelayer_normalization/add:z:01sequential/dense/Tensordot/Reshape/shape:output:0*
T0* 
_output_shapes
:
NŹś
!sequential/dense/Tensordot/MatMulMatMul+sequential/dense/Tensordot/Reshape:output:01sequential/dense/Tensordot/ReadVariableOp:value:0*
T0* 
_output_shapes
:
NŹu
 sequential/dense/Tensordot/shapeConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ,  Ť
sequential/dense/TensordotReshape+sequential/dense/Tensordot/MatMul:product:0)sequential/dense/Tensordot/shape:output:0*
T0*#
_output_shapes
:ddŹ
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes	
:Ź*
dtype0§
sequential/dense/BiasAddBiasAdd#sequential/dense/Tensordot:output:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*#
_output_shapes
:ddŹo
sequential/dense/LeakyRelu	LeakyRelu!sequential/dense/BiasAdd:output:0*#
_output_shapes
:ddŹ˘
+sequential/dense_1/Tensordot/ReadVariableOpReadVariableOp4sequential_dense_1_tensordot_readvariableop_resource* 
_output_shapes
:
ŹŹ*
dtype0{
*sequential/dense_1/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"'  ,  š
$sequential/dense_1/Tensordot/ReshapeReshape(sequential/dense/LeakyRelu:activations:03sequential/dense_1/Tensordot/Reshape/shape:output:0*
T0* 
_output_shapes
:
NŹź
#sequential/dense_1/Tensordot/MatMulMatMul-sequential/dense_1/Tensordot/Reshape:output:03sequential/dense_1/Tensordot/ReadVariableOp:value:0*
T0* 
_output_shapes
:
NŹw
"sequential/dense_1/Tensordot/shapeConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ,  ą
sequential/dense_1/TensordotReshape-sequential/dense_1/Tensordot/MatMul:product:0+sequential/dense_1/Tensordot/shape:output:0*
T0*#
_output_shapes
:ddŹ
)sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:Ź*
dtype0­
sequential/dense_1/BiasAddBiasAdd%sequential/dense_1/Tensordot:output:01sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*#
_output_shapes
:ddŹ~
add_1AddV2#sequential/dense_1/BiasAdd:output:0layer_normalization/add:z:0*
T0*#
_output_shapes
:ddŹp
layer_normalization/Shape_1Const*
_output_shapes
:*
dtype0*!
valueB"d   d   ,  s
)layer_normalization/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+layer_normalization/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+layer_normalization/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ż
#layer_normalization/strided_slice_3StridedSlice$layer_normalization/Shape_1:output:02layer_normalization/strided_slice_3/stack:output:04layer_normalization/strided_slice_3/stack_1:output:04layer_normalization/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
layer_normalization/mul_4/xConst*
_output_shapes
: *
dtype0*
value	B :
layer_normalization/mul_4Mul$layer_normalization/mul_4/x:output:0,layer_normalization/strided_slice_3:output:0*
T0*
_output_shapes
: s
)layer_normalization/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:u
+layer_normalization/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+layer_normalization/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ż
#layer_normalization/strided_slice_4StridedSlice$layer_normalization/Shape_1:output:02layer_normalization/strided_slice_4/stack:output:04layer_normalization/strided_slice_4/stack_1:output:04layer_normalization/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
layer_normalization/mul_5Mullayer_normalization/mul_4:z:0,layer_normalization/strided_slice_4:output:0*
T0*
_output_shapes
: s
)layer_normalization/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:u
+layer_normalization/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+layer_normalization/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ż
#layer_normalization/strided_slice_5StridedSlice$layer_normalization/Shape_1:output:02layer_normalization/strided_slice_5/stack:output:04layer_normalization/strided_slice_5/stack_1:output:04layer_normalization/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
layer_normalization/mul_6/xConst*
_output_shapes
: *
dtype0*
value	B :
layer_normalization/mul_6Mul$layer_normalization/mul_6/x:output:0,layer_normalization/strided_slice_5:output:0*
T0*
_output_shapes
: g
%layer_normalization/Reshape_2/shape/0Const*
_output_shapes
: *
dtype0*
value	B :g
%layer_normalization/Reshape_2/shape/3Const*
_output_shapes
: *
dtype0*
value	B :÷
#layer_normalization/Reshape_2/shapePack.layer_normalization/Reshape_2/shape/0:output:0layer_normalization/mul_5:z:0layer_normalization/mul_6:z:0.layer_normalization/Reshape_2/shape/3:output:0*
N*
T0*
_output_shapes
:
layer_normalization/Reshape_2Reshape	add_1:z:0,layer_normalization/Reshape_2/shape:output:0*
T0*(
_output_shapes
:NŹv
!layer_normalization/ones_1/packedPacklayer_normalization/mul_5:z:0*
N*
T0*
_output_shapes
:e
 layer_normalization/ones_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
layer_normalization/ones_1Fill*layer_normalization/ones_1/packed:output:0)layer_normalization/ones_1/Const:output:0*
T0*
_output_shapes	
:Nw
"layer_normalization/zeros_1/packedPacklayer_normalization/mul_5:z:0*
N*
T0*
_output_shapes
:f
!layer_normalization/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ˘
layer_normalization/zeros_1Fill+layer_normalization/zeros_1/packed:output:0*layer_normalization/zeros_1/Const:output:0*
T0*
_output_shapes	
:N^
layer_normalization/Const_2Const*
_output_shapes
: *
dtype0*
valueB ^
layer_normalization/Const_3Const*
_output_shapes
: *
dtype0*
valueB ţ
&layer_normalization/FusedBatchNormV3_1FusedBatchNormV3&layer_normalization/Reshape_2:output:0#layer_normalization/ones_1:output:0$layer_normalization/zeros_1:output:0$layer_normalization/Const_2:output:0$layer_normalization/Const_3:output:0*
T0*
U0*H
_output_shapes6
4:NŹ:N:N:N:N:*
data_formatNCHW*
epsilon%o:¨
layer_normalization/Reshape_3Reshape*layer_normalization/FusedBatchNormV3_1:y:0$layer_normalization/Shape_1:output:0*
T0*#
_output_shapes
:ddŹ
)layer_normalization/Cast_2/ReadVariableOpReadVariableOp0layer_normalization_cast_readvariableop_resource*
_output_shapes	
:Ź*
dtype0Š
layer_normalization/mul_7Mul&layer_normalization/Reshape_3:output:01layer_normalization/Cast_2/ReadVariableOp:value:0*
T0*#
_output_shapes
:ddŹ
)layer_normalization/Cast_3/ReadVariableOpReadVariableOp2layer_normalization_cast_1_readvariableop_resource*
_output_shapes	
:Ź*
dtype0˘
layer_normalization/add_1AddV2layer_normalization/mul_7:z:01layer_normalization/Cast_3/ReadVariableOp:value:0*
T0*#
_output_shapes
:ddŹh
IdentityIdentitylayer_normalization/add_1:z:0^NoOp*
T0*#
_output_shapes
:ddŹÍ
NoOpNoOp'^attention_head/StatefulPartitionedCall(^layer_normalization/Cast/ReadVariableOp*^layer_normalization/Cast_1/ReadVariableOp*^layer_normalization/Cast_2/ReadVariableOp*^layer_normalization/Cast_3/ReadVariableOp(^sequential/dense/BiasAdd/ReadVariableOp*^sequential/dense/Tensordot/ReadVariableOp*^sequential/dense_1/BiasAdd/ReadVariableOp,^sequential/dense_1/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:	dŹ: : : : : : : : : 2P
&attention_head/StatefulPartitionedCall&attention_head/StatefulPartitionedCall2R
'layer_normalization/Cast/ReadVariableOp'layer_normalization/Cast/ReadVariableOp2V
)layer_normalization/Cast_1/ReadVariableOp)layer_normalization/Cast_1/ReadVariableOp2V
)layer_normalization/Cast_2/ReadVariableOp)layer_normalization/Cast_2/ReadVariableOp2V
)layer_normalization/Cast_3/ReadVariableOp)layer_normalization/Cast_3/ReadVariableOp2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2V
)sequential/dense/Tensordot/ReadVariableOp)sequential/dense/Tensordot/ReadVariableOp2V
)sequential/dense_1/BiasAdd/ReadVariableOp)sequential/dense_1/BiasAdd/ReadVariableOp2Z
+sequential/dense_1/Tensordot/ReadVariableOp+sequential/dense_1/Tensordot/ReadVariableOp:G C

_output_shapes
:	dŹ
 
_user_specified_nameinputs

R
6__inference_global_max_pooling2d_layer_call_fn_3834766

inputs
identityĹ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Z
fURS
Q__inference_global_max_pooling2d_layer_call_and_return_conditional_losses_3832389i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:r n
J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs

ă
5__inference_transformer_block_1_layer_call_fn_3834425

inputs
context_sequence
unknown:
ŹŹ
	unknown_0:
ŹŹ
	unknown_1:
ŹŹ
	unknown_2:	Ź
	unknown_3:	Ź
	unknown_4:
ŹŹ
	unknown_5:
ŹŹ
	unknown_6:
ŹŹ
	unknown_7:
ŹŹ
	unknown_8:	Ź
	unknown_9:
ŹŹ

unknown_10:	Ź
identity˘StatefulPartitionedCall÷
StatefulPartitionedCallStatefulPartitionedCallinputscontext_sequenceunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:ddŹ*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_transformer_block_1_layer_call_and_return_conditional_losses_3833131k
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*#
_output_shapes
:ddŹ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:ddŹ:	dŹ: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:ddŹ
 
_user_specified_nameinputs:QM

_output_shapes
:	dŹ
*
_user_specified_namecontext_sequence
ś
ü
B__inference_dense_layer_call_and_return_conditional_losses_3832031

inputs5
!tensordot_readvariableop_resource:
ŹŹ.
biasadd_readvariableop_resource:	Ź
identity˘BiasAdd/ReadVariableOp˘Tensordot/ReadVariableOp|
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource* 
_output_shapes
:
ŹŹ*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ť
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ż
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:z
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙dŹ
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ź\
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:ŹY
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : §
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙dŹs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:Ź*
dtype0}
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙dŹV
	LeakyRelu	LeakyReluBiasAdd:output:0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙dŹk
IdentityIdentityLeakyRelu:activations:0^NoOp*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙dŹz
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:˙˙˙˙˙˙˙˙˙dŹ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:˙˙˙˙˙˙˙˙˙dŹ
 
_user_specified_nameinputs
ş
Ś
F__inference_embedding_layer_call_and_return_conditional_losses_3832516

inputs,
embedding_lookup_3832510:
XŹ
identity˘embedding_lookup¸
embedding_lookupResourceGatherembedding_lookup_3832510inputs*
Tindices0*+
_class!
loc:@embedding_lookup/3832510*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ź*
dtype0 
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*+
_class!
loc:@embedding_lookup/3832510*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ź~
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Źt
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ŹY
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:˙˙˙˙˙˙˙˙˙: 2$
embedding_lookupembedding_lookup:K G
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
űż
Ż

__inference_call_3830201

inputs
context_sequence,
attention_head_2_3830037:
ŹŹ,
attention_head_2_3830039:
ŹŹ,
attention_head_2_3830041:
ŹŹA
2layer_normalization_1_cast_readvariableop_resource:	ŹC
4layer_normalization_1_cast_1_readvariableop_resource:	Ź,
attention_head_3_3830088:
ŹŹ,
attention_head_3_3830090:
ŹŹ,
attention_head_3_3830092:
ŹŹJ
6sequential_1_dense_2_tensordot_readvariableop_resource:
ŹŹC
4sequential_1_dense_2_biasadd_readvariableop_resource:	ŹJ
6sequential_1_dense_3_tensordot_readvariableop_resource:
ŹŹC
4sequential_1_dense_3_biasadd_readvariableop_resource:	Ź
identity˘(attention_head_2/StatefulPartitionedCall˘(attention_head_3/StatefulPartitionedCall˘)layer_normalization_1/Cast/ReadVariableOp˘+layer_normalization_1/Cast_1/ReadVariableOp˘+layer_normalization_1/Cast_2/ReadVariableOp˘+layer_normalization_1/Cast_3/ReadVariableOp˘+layer_normalization_1/Cast_4/ReadVariableOp˘+layer_normalization_1/Cast_5/ReadVariableOp˘+sequential_1/dense_2/BiasAdd/ReadVariableOp˘-sequential_1/dense_2/Tensordot/ReadVariableOp˘+sequential_1/dense_3/BiasAdd/ReadVariableOp˘-sequential_1/dense_3/Tensordot/ReadVariableOp
(attention_head_2/StatefulPartitionedCallStatefulPartitionedCallinputsinputsinputsattention_head_2_3830037attention_head_2_3830039attention_head_2_3830041*
Tin

2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:ddŹ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *!
fR
__inference_call_3827587u
addAddV21attention_head_2/StatefulPartitionedCall:output:0inputs*
T0*#
_output_shapes
:ddŹp
layer_normalization_1/ShapeConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ,  s
)layer_normalization_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+layer_normalization_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+layer_normalization_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ż
#layer_normalization_1/strided_sliceStridedSlice$layer_normalization_1/Shape:output:02layer_normalization_1/strided_slice/stack:output:04layer_normalization_1/strided_slice/stack_1:output:04layer_normalization_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
layer_normalization_1/mul/xConst*
_output_shapes
: *
dtype0*
value	B :
layer_normalization_1/mulMul$layer_normalization_1/mul/x:output:0,layer_normalization_1/strided_slice:output:0*
T0*
_output_shapes
: u
+layer_normalization_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ç
%layer_normalization_1/strided_slice_1StridedSlice$layer_normalization_1/Shape:output:04layer_normalization_1/strided_slice_1/stack:output:06layer_normalization_1/strided_slice_1/stack_1:output:06layer_normalization_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
layer_normalization_1/mul_1Mullayer_normalization_1/mul:z:0.layer_normalization_1/strided_slice_1:output:0*
T0*
_output_shapes
: u
+layer_normalization_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ç
%layer_normalization_1/strided_slice_2StridedSlice$layer_normalization_1/Shape:output:04layer_normalization_1/strided_slice_2/stack:output:06layer_normalization_1/strided_slice_2/stack_1:output:06layer_normalization_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
layer_normalization_1/mul_2/xConst*
_output_shapes
: *
dtype0*
value	B :
layer_normalization_1/mul_2Mul&layer_normalization_1/mul_2/x:output:0.layer_normalization_1/strided_slice_2:output:0*
T0*
_output_shapes
: g
%layer_normalization_1/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :g
%layer_normalization_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :ű
#layer_normalization_1/Reshape/shapePack.layer_normalization_1/Reshape/shape/0:output:0layer_normalization_1/mul_1:z:0layer_normalization_1/mul_2:z:0.layer_normalization_1/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:
layer_normalization_1/ReshapeReshapeadd:z:0,layer_normalization_1/Reshape/shape:output:0*
T0*(
_output_shapes
:NŹx
!layer_normalization_1/ones/packedPacklayer_normalization_1/mul_1:z:0*
N*
T0*
_output_shapes
:e
 layer_normalization_1/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
layer_normalization_1/onesFill*layer_normalization_1/ones/packed:output:0)layer_normalization_1/ones/Const:output:0*
T0*
_output_shapes	
:Ny
"layer_normalization_1/zeros/packedPacklayer_normalization_1/mul_1:z:0*
N*
T0*
_output_shapes
:f
!layer_normalization_1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ˘
layer_normalization_1/zerosFill+layer_normalization_1/zeros/packed:output:0*layer_normalization_1/zeros/Const:output:0*
T0*
_output_shapes	
:N^
layer_normalization_1/ConstConst*
_output_shapes
: *
dtype0*
valueB `
layer_normalization_1/Const_1Const*
_output_shapes
: *
dtype0*
valueB 
&layer_normalization_1/FusedBatchNormV3FusedBatchNormV3&layer_normalization_1/Reshape:output:0#layer_normalization_1/ones:output:0$layer_normalization_1/zeros:output:0$layer_normalization_1/Const:output:0&layer_normalization_1/Const_1:output:0*
T0*
U0*H
_output_shapes6
4:NŹ:N:N:N:N:*
data_formatNCHW*
epsilon%o:Ş
layer_normalization_1/Reshape_1Reshape*layer_normalization_1/FusedBatchNormV3:y:0$layer_normalization_1/Shape:output:0*
T0*#
_output_shapes
:ddŹ
)layer_normalization_1/Cast/ReadVariableOpReadVariableOp2layer_normalization_1_cast_readvariableop_resource*
_output_shapes	
:Ź*
dtype0­
layer_normalization_1/mul_3Mul(layer_normalization_1/Reshape_1:output:01layer_normalization_1/Cast/ReadVariableOp:value:0*
T0*#
_output_shapes
:ddŹ
+layer_normalization_1/Cast_1/ReadVariableOpReadVariableOp4layer_normalization_1_cast_1_readvariableop_resource*
_output_shapes	
:Ź*
dtype0Ś
layer_normalization_1/addAddV2layer_normalization_1/mul_3:z:03layer_normalization_1/Cast_1/ReadVariableOp:value:0*
T0*#
_output_shapes
:ddŹŔ
(attention_head_3/StatefulPartitionedCallStatefulPartitionedCalllayer_normalization_1/add:z:0layer_normalization_1/add:z:0context_sequenceattention_head_3_3830088attention_head_3_3830090attention_head_3_3830092*
Tin

2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:ddŹ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *!
fR
__inference_call_3827682
add_1AddV21attention_head_3/StatefulPartitionedCall:output:0layer_normalization_1/add:z:0*
T0*#
_output_shapes
:ddŹr
layer_normalization_1/Shape_1Const*
_output_shapes
:*
dtype0*!
valueB"d   d   ,  u
+layer_normalization_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-layer_normalization_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:É
%layer_normalization_1/strided_slice_3StridedSlice&layer_normalization_1/Shape_1:output:04layer_normalization_1/strided_slice_3/stack:output:06layer_normalization_1/strided_slice_3/stack_1:output:06layer_normalization_1/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
layer_normalization_1/mul_4/xConst*
_output_shapes
: *
dtype0*
value	B :
layer_normalization_1/mul_4Mul&layer_normalization_1/mul_4/x:output:0.layer_normalization_1/strided_slice_3:output:0*
T0*
_output_shapes
: u
+layer_normalization_1/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_1/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_1/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:É
%layer_normalization_1/strided_slice_4StridedSlice&layer_normalization_1/Shape_1:output:04layer_normalization_1/strided_slice_4/stack:output:06layer_normalization_1/strided_slice_4/stack_1:output:06layer_normalization_1/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
layer_normalization_1/mul_5Mullayer_normalization_1/mul_4:z:0.layer_normalization_1/strided_slice_4:output:0*
T0*
_output_shapes
: u
+layer_normalization_1/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_1/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_1/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:É
%layer_normalization_1/strided_slice_5StridedSlice&layer_normalization_1/Shape_1:output:04layer_normalization_1/strided_slice_5/stack:output:06layer_normalization_1/strided_slice_5/stack_1:output:06layer_normalization_1/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
layer_normalization_1/mul_6/xConst*
_output_shapes
: *
dtype0*
value	B :
layer_normalization_1/mul_6Mul&layer_normalization_1/mul_6/x:output:0.layer_normalization_1/strided_slice_5:output:0*
T0*
_output_shapes
: i
'layer_normalization_1/Reshape_2/shape/0Const*
_output_shapes
: *
dtype0*
value	B :i
'layer_normalization_1/Reshape_2/shape/3Const*
_output_shapes
: *
dtype0*
value	B :
%layer_normalization_1/Reshape_2/shapePack0layer_normalization_1/Reshape_2/shape/0:output:0layer_normalization_1/mul_5:z:0layer_normalization_1/mul_6:z:00layer_normalization_1/Reshape_2/shape/3:output:0*
N*
T0*
_output_shapes
:
layer_normalization_1/Reshape_2Reshape	add_1:z:0.layer_normalization_1/Reshape_2/shape:output:0*
T0*(
_output_shapes
:NŹz
#layer_normalization_1/ones_1/packedPacklayer_normalization_1/mul_5:z:0*
N*
T0*
_output_shapes
:g
"layer_normalization_1/ones_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ľ
layer_normalization_1/ones_1Fill,layer_normalization_1/ones_1/packed:output:0+layer_normalization_1/ones_1/Const:output:0*
T0*
_output_shapes	
:N{
$layer_normalization_1/zeros_1/packedPacklayer_normalization_1/mul_5:z:0*
N*
T0*
_output_shapes
:h
#layer_normalization_1/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ¨
layer_normalization_1/zeros_1Fill-layer_normalization_1/zeros_1/packed:output:0,layer_normalization_1/zeros_1/Const:output:0*
T0*
_output_shapes	
:N`
layer_normalization_1/Const_2Const*
_output_shapes
: *
dtype0*
valueB `
layer_normalization_1/Const_3Const*
_output_shapes
: *
dtype0*
valueB 
(layer_normalization_1/FusedBatchNormV3_1FusedBatchNormV3(layer_normalization_1/Reshape_2:output:0%layer_normalization_1/ones_1:output:0&layer_normalization_1/zeros_1:output:0&layer_normalization_1/Const_2:output:0&layer_normalization_1/Const_3:output:0*
T0*
U0*H
_output_shapes6
4:NŹ:N:N:N:N:*
data_formatNCHW*
epsilon%o:Ž
layer_normalization_1/Reshape_3Reshape,layer_normalization_1/FusedBatchNormV3_1:y:0&layer_normalization_1/Shape_1:output:0*
T0*#
_output_shapes
:ddŹ
+layer_normalization_1/Cast_2/ReadVariableOpReadVariableOp2layer_normalization_1_cast_readvariableop_resource*
_output_shapes	
:Ź*
dtype0Ż
layer_normalization_1/mul_7Mul(layer_normalization_1/Reshape_3:output:03layer_normalization_1/Cast_2/ReadVariableOp:value:0*
T0*#
_output_shapes
:ddŹ
+layer_normalization_1/Cast_3/ReadVariableOpReadVariableOp4layer_normalization_1_cast_1_readvariableop_resource*
_output_shapes	
:Ź*
dtype0¨
layer_normalization_1/add_1AddV2layer_normalization_1/mul_7:z:03layer_normalization_1/Cast_3/ReadVariableOp:value:0*
T0*#
_output_shapes
:ddŹŚ
-sequential_1/dense_2/Tensordot/ReadVariableOpReadVariableOp6sequential_1_dense_2_tensordot_readvariableop_resource* 
_output_shapes
:
ŹŹ*
dtype0}
,sequential_1/dense_2/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"'  ,  ´
&sequential_1/dense_2/Tensordot/ReshapeReshapelayer_normalization_1/add_1:z:05sequential_1/dense_2/Tensordot/Reshape/shape:output:0*
T0* 
_output_shapes
:
NŹÂ
%sequential_1/dense_2/Tensordot/MatMulMatMul/sequential_1/dense_2/Tensordot/Reshape:output:05sequential_1/dense_2/Tensordot/ReadVariableOp:value:0*
T0* 
_output_shapes
:
NŹy
$sequential_1/dense_2/Tensordot/shapeConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ,  ˇ
sequential_1/dense_2/TensordotReshape/sequential_1/dense_2/Tensordot/MatMul:product:0-sequential_1/dense_2/Tensordot/shape:output:0*
T0*#
_output_shapes
:ddŹ
+sequential_1/dense_2/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:Ź*
dtype0ł
sequential_1/dense_2/BiasAddBiasAdd'sequential_1/dense_2/Tensordot:output:03sequential_1/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*#
_output_shapes
:ddŹw
sequential_1/dense_2/LeakyRelu	LeakyRelu%sequential_1/dense_2/BiasAdd:output:0*#
_output_shapes
:ddŹŚ
-sequential_1/dense_3/Tensordot/ReadVariableOpReadVariableOp6sequential_1_dense_3_tensordot_readvariableop_resource* 
_output_shapes
:
ŹŹ*
dtype0}
,sequential_1/dense_3/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"'  ,  Á
&sequential_1/dense_3/Tensordot/ReshapeReshape,sequential_1/dense_2/LeakyRelu:activations:05sequential_1/dense_3/Tensordot/Reshape/shape:output:0*
T0* 
_output_shapes
:
NŹÂ
%sequential_1/dense_3/Tensordot/MatMulMatMul/sequential_1/dense_3/Tensordot/Reshape:output:05sequential_1/dense_3/Tensordot/ReadVariableOp:value:0*
T0* 
_output_shapes
:
NŹy
$sequential_1/dense_3/Tensordot/shapeConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ,  ˇ
sequential_1/dense_3/TensordotReshape/sequential_1/dense_3/Tensordot/MatMul:product:0-sequential_1/dense_3/Tensordot/shape:output:0*
T0*#
_output_shapes
:ddŹ
+sequential_1/dense_3/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_3_biasadd_readvariableop_resource*
_output_shapes	
:Ź*
dtype0ł
sequential_1/dense_3/BiasAddBiasAdd'sequential_1/dense_3/Tensordot:output:03sequential_1/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*#
_output_shapes
:ddŹ
add_2AddV2%sequential_1/dense_3/BiasAdd:output:0layer_normalization_1/add_1:z:0*
T0*#
_output_shapes
:ddŹr
layer_normalization_1/Shape_2Const*
_output_shapes
:*
dtype0*!
valueB"d   d   ,  u
+layer_normalization_1/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-layer_normalization_1/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_1/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB:É
%layer_normalization_1/strided_slice_6StridedSlice&layer_normalization_1/Shape_2:output:04layer_normalization_1/strided_slice_6/stack:output:06layer_normalization_1/strided_slice_6/stack_1:output:06layer_normalization_1/strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
layer_normalization_1/mul_8/xConst*
_output_shapes
: *
dtype0*
value	B :
layer_normalization_1/mul_8Mul&layer_normalization_1/mul_8/x:output:0.layer_normalization_1/strided_slice_6:output:0*
T0*
_output_shapes
: u
+layer_normalization_1/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_1/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_1/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB:É
%layer_normalization_1/strided_slice_7StridedSlice&layer_normalization_1/Shape_2:output:04layer_normalization_1/strided_slice_7/stack:output:06layer_normalization_1/strided_slice_7/stack_1:output:06layer_normalization_1/strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
layer_normalization_1/mul_9Mullayer_normalization_1/mul_8:z:0.layer_normalization_1/strided_slice_7:output:0*
T0*
_output_shapes
: u
+layer_normalization_1/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_1/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_1/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:É
%layer_normalization_1/strided_slice_8StridedSlice&layer_normalization_1/Shape_2:output:04layer_normalization_1/strided_slice_8/stack:output:06layer_normalization_1/strided_slice_8/stack_1:output:06layer_normalization_1/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask`
layer_normalization_1/mul_10/xConst*
_output_shapes
: *
dtype0*
value	B :
layer_normalization_1/mul_10Mul'layer_normalization_1/mul_10/x:output:0.layer_normalization_1/strided_slice_8:output:0*
T0*
_output_shapes
: i
'layer_normalization_1/Reshape_4/shape/0Const*
_output_shapes
: *
dtype0*
value	B :i
'layer_normalization_1/Reshape_4/shape/3Const*
_output_shapes
: *
dtype0*
value	B :
%layer_normalization_1/Reshape_4/shapePack0layer_normalization_1/Reshape_4/shape/0:output:0layer_normalization_1/mul_9:z:0 layer_normalization_1/mul_10:z:00layer_normalization_1/Reshape_4/shape/3:output:0*
N*
T0*
_output_shapes
:
layer_normalization_1/Reshape_4Reshape	add_2:z:0.layer_normalization_1/Reshape_4/shape:output:0*
T0*(
_output_shapes
:NŹz
#layer_normalization_1/ones_2/packedPacklayer_normalization_1/mul_9:z:0*
N*
T0*
_output_shapes
:g
"layer_normalization_1/ones_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ľ
layer_normalization_1/ones_2Fill,layer_normalization_1/ones_2/packed:output:0+layer_normalization_1/ones_2/Const:output:0*
T0*
_output_shapes	
:N{
$layer_normalization_1/zeros_2/packedPacklayer_normalization_1/mul_9:z:0*
N*
T0*
_output_shapes
:h
#layer_normalization_1/zeros_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ¨
layer_normalization_1/zeros_2Fill-layer_normalization_1/zeros_2/packed:output:0,layer_normalization_1/zeros_2/Const:output:0*
T0*
_output_shapes	
:N`
layer_normalization_1/Const_4Const*
_output_shapes
: *
dtype0*
valueB `
layer_normalization_1/Const_5Const*
_output_shapes
: *
dtype0*
valueB 
(layer_normalization_1/FusedBatchNormV3_2FusedBatchNormV3(layer_normalization_1/Reshape_4:output:0%layer_normalization_1/ones_2:output:0&layer_normalization_1/zeros_2:output:0&layer_normalization_1/Const_4:output:0&layer_normalization_1/Const_5:output:0*
T0*
U0*H
_output_shapes6
4:NŹ:N:N:N:N:*
data_formatNCHW*
epsilon%o:Ž
layer_normalization_1/Reshape_5Reshape,layer_normalization_1/FusedBatchNormV3_2:y:0&layer_normalization_1/Shape_2:output:0*
T0*#
_output_shapes
:ddŹ
+layer_normalization_1/Cast_4/ReadVariableOpReadVariableOp2layer_normalization_1_cast_readvariableop_resource*
_output_shapes	
:Ź*
dtype0°
layer_normalization_1/mul_11Mul(layer_normalization_1/Reshape_5:output:03layer_normalization_1/Cast_4/ReadVariableOp:value:0*
T0*#
_output_shapes
:ddŹ
+layer_normalization_1/Cast_5/ReadVariableOpReadVariableOp4layer_normalization_1_cast_1_readvariableop_resource*
_output_shapes	
:Ź*
dtype0Š
layer_normalization_1/add_2AddV2 layer_normalization_1/mul_11:z:03layer_normalization_1/Cast_5/ReadVariableOp:value:0*
T0*#
_output_shapes
:ddŹj
IdentityIdentitylayer_normalization_1/add_2:z:0^NoOp*
T0*#
_output_shapes
:ddŹę
NoOpNoOp)^attention_head_2/StatefulPartitionedCall)^attention_head_3/StatefulPartitionedCall*^layer_normalization_1/Cast/ReadVariableOp,^layer_normalization_1/Cast_1/ReadVariableOp,^layer_normalization_1/Cast_2/ReadVariableOp,^layer_normalization_1/Cast_3/ReadVariableOp,^layer_normalization_1/Cast_4/ReadVariableOp,^layer_normalization_1/Cast_5/ReadVariableOp,^sequential_1/dense_2/BiasAdd/ReadVariableOp.^sequential_1/dense_2/Tensordot/ReadVariableOp,^sequential_1/dense_3/BiasAdd/ReadVariableOp.^sequential_1/dense_3/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:ddŹ:	dŹ: : : : : : : : : : : : 2T
(attention_head_2/StatefulPartitionedCall(attention_head_2/StatefulPartitionedCall2T
(attention_head_3/StatefulPartitionedCall(attention_head_3/StatefulPartitionedCall2V
)layer_normalization_1/Cast/ReadVariableOp)layer_normalization_1/Cast/ReadVariableOp2Z
+layer_normalization_1/Cast_1/ReadVariableOp+layer_normalization_1/Cast_1/ReadVariableOp2Z
+layer_normalization_1/Cast_2/ReadVariableOp+layer_normalization_1/Cast_2/ReadVariableOp2Z
+layer_normalization_1/Cast_3/ReadVariableOp+layer_normalization_1/Cast_3/ReadVariableOp2Z
+layer_normalization_1/Cast_4/ReadVariableOp+layer_normalization_1/Cast_4/ReadVariableOp2Z
+layer_normalization_1/Cast_5/ReadVariableOp+layer_normalization_1/Cast_5/ReadVariableOp2Z
+sequential_1/dense_2/BiasAdd/ReadVariableOp+sequential_1/dense_2/BiasAdd/ReadVariableOp2^
-sequential_1/dense_2/Tensordot/ReadVariableOp-sequential_1/dense_2/Tensordot/ReadVariableOp2Z
+sequential_1/dense_3/BiasAdd/ReadVariableOp+sequential_1/dense_3/BiasAdd/ReadVariableOp2^
-sequential_1/dense_3/Tensordot/ReadVariableOp-sequential_1/dense_3/Tensordot/ReadVariableOp:K G
#
_output_shapes
:ddŹ
 
_user_specified_nameinputs:QM

_output_shapes
:	dŹ
*
_user_specified_namecontext_sequence
°
ś
I__inference_sequential_1_layer_call_and_return_conditional_losses_3832327

inputs#
dense_2_3832316:
ŹŹ
dense_2_3832318:	Ź#
dense_3_3832321:
ŹŹ
dense_3_3832323:	Ź
identity˘dense_2/StatefulPartitionedCall˘dense_3/StatefulPartitionedCallô
dense_2/StatefulPartitionedCallStatefulPartitionedCallinputsdense_2_3832316dense_2_3832318*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:˙˙˙˙˙˙˙˙˙dŹ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_2_layer_call_and_return_conditional_losses_3832224
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_3832321dense_3_3832323*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:˙˙˙˙˙˙˙˙˙dŹ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_3_layer_call_and_return_conditional_losses_3832260|
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙dŹ
NoOpNoOp ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :˙˙˙˙˙˙˙˙˙dŹ: : : : 2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall:T P
,
_output_shapes
:˙˙˙˙˙˙˙˙˙dŹ
 
_user_specified_nameinputs
ń
Ü
__inference_call_3827515

inputs*
attention_head_3827400:
ŹŹ*
attention_head_3827402:
ŹŹ*
attention_head_3827404:
ŹŹ?
0layer_normalization_cast_readvariableop_resource:	ŹA
2layer_normalization_cast_1_readvariableop_resource:	ŹF
2sequential_dense_tensordot_readvariableop_resource:
ŹŹ?
0sequential_dense_biasadd_readvariableop_resource:	ŹH
4sequential_dense_1_tensordot_readvariableop_resource:
ŹŹA
2sequential_dense_1_biasadd_readvariableop_resource:	Ź
identity˘&attention_head/StatefulPartitionedCall˘'layer_normalization/Cast/ReadVariableOp˘)layer_normalization/Cast_1/ReadVariableOp˘)layer_normalization/Cast_2/ReadVariableOp˘)layer_normalization/Cast_3/ReadVariableOp˘'sequential/dense/BiasAdd/ReadVariableOp˘)sequential/dense/Tensordot/ReadVariableOp˘)sequential/dense_1/BiasAdd/ReadVariableOp˘+sequential/dense_1/Tensordot/ReadVariableOp
&attention_head/StatefulPartitionedCallStatefulPartitionedCallinputsinputsinputsattention_head_3827400attention_head_3827402attention_head_3827404*
Tin

2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:ddŹ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *!
fR
__inference_call_3827399s
addAddV2/attention_head/StatefulPartitionedCall:output:0inputs*
T0*#
_output_shapes
:ddŹn
layer_normalization/ShapeConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ,  q
'layer_normalization/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)layer_normalization/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)layer_normalization/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ľ
!layer_normalization/strided_sliceStridedSlice"layer_normalization/Shape:output:00layer_normalization/strided_slice/stack:output:02layer_normalization/strided_slice/stack_1:output:02layer_normalization/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask[
layer_normalization/mul/xConst*
_output_shapes
: *
dtype0*
value	B :
layer_normalization/mulMul"layer_normalization/mul/x:output:0*layer_normalization/strided_slice:output:0*
T0*
_output_shapes
: s
)layer_normalization/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:u
+layer_normalization/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+layer_normalization/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:˝
#layer_normalization/strided_slice_1StridedSlice"layer_normalization/Shape:output:02layer_normalization/strided_slice_1/stack:output:04layer_normalization/strided_slice_1/stack_1:output:04layer_normalization/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
layer_normalization/mul_1Mullayer_normalization/mul:z:0,layer_normalization/strided_slice_1:output:0*
T0*
_output_shapes
: s
)layer_normalization/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:u
+layer_normalization/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+layer_normalization/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:˝
#layer_normalization/strided_slice_2StridedSlice"layer_normalization/Shape:output:02layer_normalization/strided_slice_2/stack:output:04layer_normalization/strided_slice_2/stack_1:output:04layer_normalization/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
layer_normalization/mul_2/xConst*
_output_shapes
: *
dtype0*
value	B :
layer_normalization/mul_2Mul$layer_normalization/mul_2/x:output:0,layer_normalization/strided_slice_2:output:0*
T0*
_output_shapes
: e
#layer_normalization/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :e
#layer_normalization/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :ń
!layer_normalization/Reshape/shapePack,layer_normalization/Reshape/shape/0:output:0layer_normalization/mul_1:z:0layer_normalization/mul_2:z:0,layer_normalization/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:
layer_normalization/ReshapeReshapeadd:z:0*layer_normalization/Reshape/shape:output:0*
T0*(
_output_shapes
:NŹt
layer_normalization/ones/packedPacklayer_normalization/mul_1:z:0*
N*
T0*
_output_shapes
:c
layer_normalization/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
layer_normalization/onesFill(layer_normalization/ones/packed:output:0'layer_normalization/ones/Const:output:0*
T0*
_output_shapes	
:Nu
 layer_normalization/zeros/packedPacklayer_normalization/mul_1:z:0*
N*
T0*
_output_shapes
:d
layer_normalization/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
layer_normalization/zerosFill)layer_normalization/zeros/packed:output:0(layer_normalization/zeros/Const:output:0*
T0*
_output_shapes	
:N\
layer_normalization/ConstConst*
_output_shapes
: *
dtype0*
valueB ^
layer_normalization/Const_1Const*
_output_shapes
: *
dtype0*
valueB ô
$layer_normalization/FusedBatchNormV3FusedBatchNormV3$layer_normalization/Reshape:output:0!layer_normalization/ones:output:0"layer_normalization/zeros:output:0"layer_normalization/Const:output:0$layer_normalization/Const_1:output:0*
T0*
U0*H
_output_shapes6
4:NŹ:N:N:N:N:*
data_formatNCHW*
epsilon%o:¤
layer_normalization/Reshape_1Reshape(layer_normalization/FusedBatchNormV3:y:0"layer_normalization/Shape:output:0*
T0*#
_output_shapes
:ddŹ
'layer_normalization/Cast/ReadVariableOpReadVariableOp0layer_normalization_cast_readvariableop_resource*
_output_shapes	
:Ź*
dtype0§
layer_normalization/mul_3Mul&layer_normalization/Reshape_1:output:0/layer_normalization/Cast/ReadVariableOp:value:0*
T0*#
_output_shapes
:ddŹ
)layer_normalization/Cast_1/ReadVariableOpReadVariableOp2layer_normalization_cast_1_readvariableop_resource*
_output_shapes	
:Ź*
dtype0 
layer_normalization/addAddV2layer_normalization/mul_3:z:01layer_normalization/Cast_1/ReadVariableOp:value:0*
T0*#
_output_shapes
:ddŹ
)sequential/dense/Tensordot/ReadVariableOpReadVariableOp2sequential_dense_tensordot_readvariableop_resource* 
_output_shapes
:
ŹŹ*
dtype0y
(sequential/dense/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"'  ,  ¨
"sequential/dense/Tensordot/ReshapeReshapelayer_normalization/add:z:01sequential/dense/Tensordot/Reshape/shape:output:0*
T0* 
_output_shapes
:
NŹś
!sequential/dense/Tensordot/MatMulMatMul+sequential/dense/Tensordot/Reshape:output:01sequential/dense/Tensordot/ReadVariableOp:value:0*
T0* 
_output_shapes
:
NŹu
 sequential/dense/Tensordot/shapeConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ,  Ť
sequential/dense/TensordotReshape+sequential/dense/Tensordot/MatMul:product:0)sequential/dense/Tensordot/shape:output:0*
T0*#
_output_shapes
:ddŹ
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes	
:Ź*
dtype0§
sequential/dense/BiasAddBiasAdd#sequential/dense/Tensordot:output:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*#
_output_shapes
:ddŹo
sequential/dense/LeakyRelu	LeakyRelu!sequential/dense/BiasAdd:output:0*#
_output_shapes
:ddŹ˘
+sequential/dense_1/Tensordot/ReadVariableOpReadVariableOp4sequential_dense_1_tensordot_readvariableop_resource* 
_output_shapes
:
ŹŹ*
dtype0{
*sequential/dense_1/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"'  ,  š
$sequential/dense_1/Tensordot/ReshapeReshape(sequential/dense/LeakyRelu:activations:03sequential/dense_1/Tensordot/Reshape/shape:output:0*
T0* 
_output_shapes
:
NŹź
#sequential/dense_1/Tensordot/MatMulMatMul-sequential/dense_1/Tensordot/Reshape:output:03sequential/dense_1/Tensordot/ReadVariableOp:value:0*
T0* 
_output_shapes
:
NŹw
"sequential/dense_1/Tensordot/shapeConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ,  ą
sequential/dense_1/TensordotReshape-sequential/dense_1/Tensordot/MatMul:product:0+sequential/dense_1/Tensordot/shape:output:0*
T0*#
_output_shapes
:ddŹ
)sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:Ź*
dtype0­
sequential/dense_1/BiasAddBiasAdd%sequential/dense_1/Tensordot:output:01sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*#
_output_shapes
:ddŹ~
add_1AddV2#sequential/dense_1/BiasAdd:output:0layer_normalization/add:z:0*
T0*#
_output_shapes
:ddŹp
layer_normalization/Shape_1Const*
_output_shapes
:*
dtype0*!
valueB"d   d   ,  s
)layer_normalization/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+layer_normalization/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+layer_normalization/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ż
#layer_normalization/strided_slice_3StridedSlice$layer_normalization/Shape_1:output:02layer_normalization/strided_slice_3/stack:output:04layer_normalization/strided_slice_3/stack_1:output:04layer_normalization/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
layer_normalization/mul_4/xConst*
_output_shapes
: *
dtype0*
value	B :
layer_normalization/mul_4Mul$layer_normalization/mul_4/x:output:0,layer_normalization/strided_slice_3:output:0*
T0*
_output_shapes
: s
)layer_normalization/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:u
+layer_normalization/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+layer_normalization/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ż
#layer_normalization/strided_slice_4StridedSlice$layer_normalization/Shape_1:output:02layer_normalization/strided_slice_4/stack:output:04layer_normalization/strided_slice_4/stack_1:output:04layer_normalization/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
layer_normalization/mul_5Mullayer_normalization/mul_4:z:0,layer_normalization/strided_slice_4:output:0*
T0*
_output_shapes
: s
)layer_normalization/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:u
+layer_normalization/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+layer_normalization/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ż
#layer_normalization/strided_slice_5StridedSlice$layer_normalization/Shape_1:output:02layer_normalization/strided_slice_5/stack:output:04layer_normalization/strided_slice_5/stack_1:output:04layer_normalization/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
layer_normalization/mul_6/xConst*
_output_shapes
: *
dtype0*
value	B :
layer_normalization/mul_6Mul$layer_normalization/mul_6/x:output:0,layer_normalization/strided_slice_5:output:0*
T0*
_output_shapes
: g
%layer_normalization/Reshape_2/shape/0Const*
_output_shapes
: *
dtype0*
value	B :g
%layer_normalization/Reshape_2/shape/3Const*
_output_shapes
: *
dtype0*
value	B :÷
#layer_normalization/Reshape_2/shapePack.layer_normalization/Reshape_2/shape/0:output:0layer_normalization/mul_5:z:0layer_normalization/mul_6:z:0.layer_normalization/Reshape_2/shape/3:output:0*
N*
T0*
_output_shapes
:
layer_normalization/Reshape_2Reshape	add_1:z:0,layer_normalization/Reshape_2/shape:output:0*
T0*(
_output_shapes
:NŹv
!layer_normalization/ones_1/packedPacklayer_normalization/mul_5:z:0*
N*
T0*
_output_shapes
:e
 layer_normalization/ones_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
layer_normalization/ones_1Fill*layer_normalization/ones_1/packed:output:0)layer_normalization/ones_1/Const:output:0*
T0*
_output_shapes	
:Nw
"layer_normalization/zeros_1/packedPacklayer_normalization/mul_5:z:0*
N*
T0*
_output_shapes
:f
!layer_normalization/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ˘
layer_normalization/zeros_1Fill+layer_normalization/zeros_1/packed:output:0*layer_normalization/zeros_1/Const:output:0*
T0*
_output_shapes	
:N^
layer_normalization/Const_2Const*
_output_shapes
: *
dtype0*
valueB ^
layer_normalization/Const_3Const*
_output_shapes
: *
dtype0*
valueB ţ
&layer_normalization/FusedBatchNormV3_1FusedBatchNormV3&layer_normalization/Reshape_2:output:0#layer_normalization/ones_1:output:0$layer_normalization/zeros_1:output:0$layer_normalization/Const_2:output:0$layer_normalization/Const_3:output:0*
T0*
U0*H
_output_shapes6
4:NŹ:N:N:N:N:*
data_formatNCHW*
epsilon%o:¨
layer_normalization/Reshape_3Reshape*layer_normalization/FusedBatchNormV3_1:y:0$layer_normalization/Shape_1:output:0*
T0*#
_output_shapes
:ddŹ
)layer_normalization/Cast_2/ReadVariableOpReadVariableOp0layer_normalization_cast_readvariableop_resource*
_output_shapes	
:Ź*
dtype0Š
layer_normalization/mul_7Mul&layer_normalization/Reshape_3:output:01layer_normalization/Cast_2/ReadVariableOp:value:0*
T0*#
_output_shapes
:ddŹ
)layer_normalization/Cast_3/ReadVariableOpReadVariableOp2layer_normalization_cast_1_readvariableop_resource*
_output_shapes	
:Ź*
dtype0˘
layer_normalization/add_1AddV2layer_normalization/mul_7:z:01layer_normalization/Cast_3/ReadVariableOp:value:0*
T0*#
_output_shapes
:ddŹh
IdentityIdentitylayer_normalization/add_1:z:0^NoOp*
T0*#
_output_shapes
:ddŹÍ
NoOpNoOp'^attention_head/StatefulPartitionedCall(^layer_normalization/Cast/ReadVariableOp*^layer_normalization/Cast_1/ReadVariableOp*^layer_normalization/Cast_2/ReadVariableOp*^layer_normalization/Cast_3/ReadVariableOp(^sequential/dense/BiasAdd/ReadVariableOp*^sequential/dense/Tensordot/ReadVariableOp*^sequential/dense_1/BiasAdd/ReadVariableOp,^sequential/dense_1/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:	dŹ: : : : : : : : : 2P
&attention_head/StatefulPartitionedCall&attention_head/StatefulPartitionedCall2R
'layer_normalization/Cast/ReadVariableOp'layer_normalization/Cast/ReadVariableOp2V
)layer_normalization/Cast_1/ReadVariableOp)layer_normalization/Cast_1/ReadVariableOp2V
)layer_normalization/Cast_2/ReadVariableOp)layer_normalization/Cast_2/ReadVariableOp2V
)layer_normalization/Cast_3/ReadVariableOp)layer_normalization/Cast_3/ReadVariableOp2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2V
)sequential/dense/Tensordot/ReadVariableOp)sequential/dense/Tensordot/ReadVariableOp2V
)sequential/dense_1/BiasAdd/ReadVariableOp)sequential/dense_1/BiasAdd/ReadVariableOp2Z
+sequential/dense_1/Tensordot/ReadVariableOp+sequential/dense_1/Tensordot/ReadVariableOp:G C

_output_shapes
:	dŹ
 
_user_specified_nameinputs
§

N__inference_transformer_block_layer_call_and_return_conditional_losses_3832646

inputs*
attention_head_3832531:
ŹŹ*
attention_head_3832533:
ŹŹ*
attention_head_3832535:
ŹŹ?
0layer_normalization_cast_readvariableop_resource:	ŹA
2layer_normalization_cast_1_readvariableop_resource:	ŹF
2sequential_dense_tensordot_readvariableop_resource:
ŹŹ?
0sequential_dense_biasadd_readvariableop_resource:	ŹH
4sequential_dense_1_tensordot_readvariableop_resource:
ŹŹA
2sequential_dense_1_biasadd_readvariableop_resource:	Ź
identity˘&attention_head/StatefulPartitionedCall˘'layer_normalization/Cast/ReadVariableOp˘)layer_normalization/Cast_1/ReadVariableOp˘)layer_normalization/Cast_2/ReadVariableOp˘)layer_normalization/Cast_3/ReadVariableOp˘'sequential/dense/BiasAdd/ReadVariableOp˘)sequential/dense/Tensordot/ReadVariableOp˘)sequential/dense_1/BiasAdd/ReadVariableOp˘+sequential/dense_1/Tensordot/ReadVariableOp
&attention_head/StatefulPartitionedCallStatefulPartitionedCallinputsinputsinputsattention_head_3832531attention_head_3832533attention_head_3832535*
Tin

2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:ddŹ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *!
fR
__inference_call_3827399s
addAddV2/attention_head/StatefulPartitionedCall:output:0inputs*
T0*#
_output_shapes
:ddŹn
layer_normalization/ShapeConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ,  q
'layer_normalization/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)layer_normalization/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)layer_normalization/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ľ
!layer_normalization/strided_sliceStridedSlice"layer_normalization/Shape:output:00layer_normalization/strided_slice/stack:output:02layer_normalization/strided_slice/stack_1:output:02layer_normalization/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask[
layer_normalization/mul/xConst*
_output_shapes
: *
dtype0*
value	B :
layer_normalization/mulMul"layer_normalization/mul/x:output:0*layer_normalization/strided_slice:output:0*
T0*
_output_shapes
: s
)layer_normalization/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:u
+layer_normalization/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+layer_normalization/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:˝
#layer_normalization/strided_slice_1StridedSlice"layer_normalization/Shape:output:02layer_normalization/strided_slice_1/stack:output:04layer_normalization/strided_slice_1/stack_1:output:04layer_normalization/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
layer_normalization/mul_1Mullayer_normalization/mul:z:0,layer_normalization/strided_slice_1:output:0*
T0*
_output_shapes
: s
)layer_normalization/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:u
+layer_normalization/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+layer_normalization/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:˝
#layer_normalization/strided_slice_2StridedSlice"layer_normalization/Shape:output:02layer_normalization/strided_slice_2/stack:output:04layer_normalization/strided_slice_2/stack_1:output:04layer_normalization/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
layer_normalization/mul_2/xConst*
_output_shapes
: *
dtype0*
value	B :
layer_normalization/mul_2Mul$layer_normalization/mul_2/x:output:0,layer_normalization/strided_slice_2:output:0*
T0*
_output_shapes
: e
#layer_normalization/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :e
#layer_normalization/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :ń
!layer_normalization/Reshape/shapePack,layer_normalization/Reshape/shape/0:output:0layer_normalization/mul_1:z:0layer_normalization/mul_2:z:0,layer_normalization/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:
layer_normalization/ReshapeReshapeadd:z:0*layer_normalization/Reshape/shape:output:0*
T0*(
_output_shapes
:NŹt
layer_normalization/ones/packedPacklayer_normalization/mul_1:z:0*
N*
T0*
_output_shapes
:c
layer_normalization/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
layer_normalization/onesFill(layer_normalization/ones/packed:output:0'layer_normalization/ones/Const:output:0*
T0*
_output_shapes	
:Nu
 layer_normalization/zeros/packedPacklayer_normalization/mul_1:z:0*
N*
T0*
_output_shapes
:d
layer_normalization/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
layer_normalization/zerosFill)layer_normalization/zeros/packed:output:0(layer_normalization/zeros/Const:output:0*
T0*
_output_shapes	
:N\
layer_normalization/ConstConst*
_output_shapes
: *
dtype0*
valueB ^
layer_normalization/Const_1Const*
_output_shapes
: *
dtype0*
valueB ô
$layer_normalization/FusedBatchNormV3FusedBatchNormV3$layer_normalization/Reshape:output:0!layer_normalization/ones:output:0"layer_normalization/zeros:output:0"layer_normalization/Const:output:0$layer_normalization/Const_1:output:0*
T0*
U0*H
_output_shapes6
4:NŹ:N:N:N:N:*
data_formatNCHW*
epsilon%o:¤
layer_normalization/Reshape_1Reshape(layer_normalization/FusedBatchNormV3:y:0"layer_normalization/Shape:output:0*
T0*#
_output_shapes
:ddŹ
'layer_normalization/Cast/ReadVariableOpReadVariableOp0layer_normalization_cast_readvariableop_resource*
_output_shapes	
:Ź*
dtype0§
layer_normalization/mul_3Mul&layer_normalization/Reshape_1:output:0/layer_normalization/Cast/ReadVariableOp:value:0*
T0*#
_output_shapes
:ddŹ
)layer_normalization/Cast_1/ReadVariableOpReadVariableOp2layer_normalization_cast_1_readvariableop_resource*
_output_shapes	
:Ź*
dtype0 
layer_normalization/addAddV2layer_normalization/mul_3:z:01layer_normalization/Cast_1/ReadVariableOp:value:0*
T0*#
_output_shapes
:ddŹ
)sequential/dense/Tensordot/ReadVariableOpReadVariableOp2sequential_dense_tensordot_readvariableop_resource* 
_output_shapes
:
ŹŹ*
dtype0y
(sequential/dense/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"'  ,  ¨
"sequential/dense/Tensordot/ReshapeReshapelayer_normalization/add:z:01sequential/dense/Tensordot/Reshape/shape:output:0*
T0* 
_output_shapes
:
NŹś
!sequential/dense/Tensordot/MatMulMatMul+sequential/dense/Tensordot/Reshape:output:01sequential/dense/Tensordot/ReadVariableOp:value:0*
T0* 
_output_shapes
:
NŹu
 sequential/dense/Tensordot/shapeConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ,  Ť
sequential/dense/TensordotReshape+sequential/dense/Tensordot/MatMul:product:0)sequential/dense/Tensordot/shape:output:0*
T0*#
_output_shapes
:ddŹ
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes	
:Ź*
dtype0§
sequential/dense/BiasAddBiasAdd#sequential/dense/Tensordot:output:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*#
_output_shapes
:ddŹo
sequential/dense/LeakyRelu	LeakyRelu!sequential/dense/BiasAdd:output:0*#
_output_shapes
:ddŹ˘
+sequential/dense_1/Tensordot/ReadVariableOpReadVariableOp4sequential_dense_1_tensordot_readvariableop_resource* 
_output_shapes
:
ŹŹ*
dtype0{
*sequential/dense_1/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"'  ,  š
$sequential/dense_1/Tensordot/ReshapeReshape(sequential/dense/LeakyRelu:activations:03sequential/dense_1/Tensordot/Reshape/shape:output:0*
T0* 
_output_shapes
:
NŹź
#sequential/dense_1/Tensordot/MatMulMatMul-sequential/dense_1/Tensordot/Reshape:output:03sequential/dense_1/Tensordot/ReadVariableOp:value:0*
T0* 
_output_shapes
:
NŹw
"sequential/dense_1/Tensordot/shapeConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ,  ą
sequential/dense_1/TensordotReshape-sequential/dense_1/Tensordot/MatMul:product:0+sequential/dense_1/Tensordot/shape:output:0*
T0*#
_output_shapes
:ddŹ
)sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:Ź*
dtype0­
sequential/dense_1/BiasAddBiasAdd%sequential/dense_1/Tensordot:output:01sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*#
_output_shapes
:ddŹ~
add_1AddV2#sequential/dense_1/BiasAdd:output:0layer_normalization/add:z:0*
T0*#
_output_shapes
:ddŹp
layer_normalization/Shape_1Const*
_output_shapes
:*
dtype0*!
valueB"d   d   ,  s
)layer_normalization/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+layer_normalization/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+layer_normalization/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ż
#layer_normalization/strided_slice_3StridedSlice$layer_normalization/Shape_1:output:02layer_normalization/strided_slice_3/stack:output:04layer_normalization/strided_slice_3/stack_1:output:04layer_normalization/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
layer_normalization/mul_4/xConst*
_output_shapes
: *
dtype0*
value	B :
layer_normalization/mul_4Mul$layer_normalization/mul_4/x:output:0,layer_normalization/strided_slice_3:output:0*
T0*
_output_shapes
: s
)layer_normalization/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:u
+layer_normalization/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+layer_normalization/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ż
#layer_normalization/strided_slice_4StridedSlice$layer_normalization/Shape_1:output:02layer_normalization/strided_slice_4/stack:output:04layer_normalization/strided_slice_4/stack_1:output:04layer_normalization/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
layer_normalization/mul_5Mullayer_normalization/mul_4:z:0,layer_normalization/strided_slice_4:output:0*
T0*
_output_shapes
: s
)layer_normalization/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:u
+layer_normalization/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+layer_normalization/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ż
#layer_normalization/strided_slice_5StridedSlice$layer_normalization/Shape_1:output:02layer_normalization/strided_slice_5/stack:output:04layer_normalization/strided_slice_5/stack_1:output:04layer_normalization/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
layer_normalization/mul_6/xConst*
_output_shapes
: *
dtype0*
value	B :
layer_normalization/mul_6Mul$layer_normalization/mul_6/x:output:0,layer_normalization/strided_slice_5:output:0*
T0*
_output_shapes
: g
%layer_normalization/Reshape_2/shape/0Const*
_output_shapes
: *
dtype0*
value	B :g
%layer_normalization/Reshape_2/shape/3Const*
_output_shapes
: *
dtype0*
value	B :÷
#layer_normalization/Reshape_2/shapePack.layer_normalization/Reshape_2/shape/0:output:0layer_normalization/mul_5:z:0layer_normalization/mul_6:z:0.layer_normalization/Reshape_2/shape/3:output:0*
N*
T0*
_output_shapes
:
layer_normalization/Reshape_2Reshape	add_1:z:0,layer_normalization/Reshape_2/shape:output:0*
T0*(
_output_shapes
:NŹv
!layer_normalization/ones_1/packedPacklayer_normalization/mul_5:z:0*
N*
T0*
_output_shapes
:e
 layer_normalization/ones_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
layer_normalization/ones_1Fill*layer_normalization/ones_1/packed:output:0)layer_normalization/ones_1/Const:output:0*
T0*
_output_shapes	
:Nw
"layer_normalization/zeros_1/packedPacklayer_normalization/mul_5:z:0*
N*
T0*
_output_shapes
:f
!layer_normalization/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ˘
layer_normalization/zeros_1Fill+layer_normalization/zeros_1/packed:output:0*layer_normalization/zeros_1/Const:output:0*
T0*
_output_shapes	
:N^
layer_normalization/Const_2Const*
_output_shapes
: *
dtype0*
valueB ^
layer_normalization/Const_3Const*
_output_shapes
: *
dtype0*
valueB ţ
&layer_normalization/FusedBatchNormV3_1FusedBatchNormV3&layer_normalization/Reshape_2:output:0#layer_normalization/ones_1:output:0$layer_normalization/zeros_1:output:0$layer_normalization/Const_2:output:0$layer_normalization/Const_3:output:0*
T0*
U0*H
_output_shapes6
4:NŹ:N:N:N:N:*
data_formatNCHW*
epsilon%o:¨
layer_normalization/Reshape_3Reshape*layer_normalization/FusedBatchNormV3_1:y:0$layer_normalization/Shape_1:output:0*
T0*#
_output_shapes
:ddŹ
)layer_normalization/Cast_2/ReadVariableOpReadVariableOp0layer_normalization_cast_readvariableop_resource*
_output_shapes	
:Ź*
dtype0Š
layer_normalization/mul_7Mul&layer_normalization/Reshape_3:output:01layer_normalization/Cast_2/ReadVariableOp:value:0*
T0*#
_output_shapes
:ddŹ
)layer_normalization/Cast_3/ReadVariableOpReadVariableOp2layer_normalization_cast_1_readvariableop_resource*
_output_shapes	
:Ź*
dtype0˘
layer_normalization/add_1AddV2layer_normalization/mul_7:z:01layer_normalization/Cast_3/ReadVariableOp:value:0*
T0*#
_output_shapes
:ddŹh
IdentityIdentitylayer_normalization/add_1:z:0^NoOp*
T0*#
_output_shapes
:ddŹÍ
NoOpNoOp'^attention_head/StatefulPartitionedCall(^layer_normalization/Cast/ReadVariableOp*^layer_normalization/Cast_1/ReadVariableOp*^layer_normalization/Cast_2/ReadVariableOp*^layer_normalization/Cast_3/ReadVariableOp(^sequential/dense/BiasAdd/ReadVariableOp*^sequential/dense/Tensordot/ReadVariableOp*^sequential/dense_1/BiasAdd/ReadVariableOp,^sequential/dense_1/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:	dŹ: : : : : : : : : 2P
&attention_head/StatefulPartitionedCall&attention_head/StatefulPartitionedCall2R
'layer_normalization/Cast/ReadVariableOp'layer_normalization/Cast/ReadVariableOp2V
)layer_normalization/Cast_1/ReadVariableOp)layer_normalization/Cast_1/ReadVariableOp2V
)layer_normalization/Cast_2/ReadVariableOp)layer_normalization/Cast_2/ReadVariableOp2V
)layer_normalization/Cast_3/ReadVariableOp)layer_normalization/Cast_3/ReadVariableOp2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2V
)sequential/dense/Tensordot/ReadVariableOp)sequential/dense/Tensordot/ReadVariableOp2V
)sequential/dense_1/BiasAdd/ReadVariableOp)sequential/dense_1/BiasAdd/ReadVariableOp2Z
+sequential/dense_1/Tensordot/ReadVariableOp+sequential/dense_1/Tensordot/ReadVariableOp:G C

_output_shapes
:	dŹ
 
_user_specified_nameinputs

E
)__inference_dropout_layer_call_fn_3834817

inputs
identityŤ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:ddŹ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_layer_call_and_return_conditional_losses_3832671\
IdentityIdentityPartitionedCall:output:0*
T0*#
_output_shapes
:ddŹ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*"
_input_shapes
:ddŹ:K G
#
_output_shapes
:ddŹ
 
_user_specified_nameinputs
Ů
ţ
D__inference_dense_1_layer_call_and_return_conditional_losses_3832067

inputs5
!tensordot_readvariableop_resource:
ŹŹ.
biasadd_readvariableop_resource:	Ź
identity˘BiasAdd/ReadVariableOp˘Tensordot/ReadVariableOp|
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource* 
_output_shapes
:
ŹŹ*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ť
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ż
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:z
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙dŹ
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ź\
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:ŹY
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : §
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙dŹs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:Ź*
dtype0}
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙dŹd
IdentityIdentityBiasAdd:output:0^NoOp*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙dŹz
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:˙˙˙˙˙˙˙˙˙dŹ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:˙˙˙˙˙˙˙˙˙dŹ
 
_user_specified_nameinputs
Ĺ
˝
I__inference_sequential_1_layer_call_and_return_conditional_losses_3832365
dense_2_input#
dense_2_3832354:
ŹŹ
dense_2_3832356:	Ź#
dense_3_3832359:
ŹŹ
dense_3_3832361:	Ź
identity˘dense_2/StatefulPartitionedCall˘dense_3/StatefulPartitionedCallű
dense_2/StatefulPartitionedCallStatefulPartitionedCalldense_2_inputdense_2_3832354dense_2_3832356*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:˙˙˙˙˙˙˙˙˙dŹ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_2_layer_call_and_return_conditional_losses_3832224
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_3832359dense_3_3832361*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:˙˙˙˙˙˙˙˙˙dŹ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_3_layer_call_and_return_conditional_losses_3832260|
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙dŹ
NoOpNoOp ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :˙˙˙˙˙˙˙˙˙dŹ: : : : 2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall:[ W
,
_output_shapes
:˙˙˙˙˙˙˙˙˙dŹ
'
_user_specified_namedense_2_input
ď=
Ô
G__inference_sequential_layer_call_and_return_conditional_losses_3834922

inputs;
'dense_tensordot_readvariableop_resource:
ŹŹ4
%dense_biasadd_readvariableop_resource:	Ź=
)dense_1_tensordot_readvariableop_resource:
ŹŹ6
'dense_1_biasadd_readvariableop_resource:	Ź
identity˘dense/BiasAdd/ReadVariableOp˘dense/Tensordot/ReadVariableOp˘dense_1/BiasAdd/ReadVariableOp˘ dense_1/Tensordot/ReadVariableOp
dense/Tensordot/ReadVariableOpReadVariableOp'dense_tensordot_readvariableop_resource* 
_output_shapes
:
ŹŹ*
dtype0^
dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:e
dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       K
dense/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:_
dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : Ó
dense/Tensordot/GatherV2GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/free:output:0&dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ×
dense/Tensordot/GatherV2_1GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/axes:output:0(dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:_
dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
dense/Tensordot/ProdProd!dense/Tensordot/GatherV2:output:0dense/Tensordot/Const:output:0*
T0*
_output_shapes
: a
dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
dense/Tensordot/Prod_1Prod#dense/Tensordot/GatherV2_1:output:0 dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: ]
dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ´
dense/Tensordot/concatConcatV2dense/Tensordot/free:output:0dense/Tensordot/axes:output:0$dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:
dense/Tensordot/stackPackdense/Tensordot/Prod:output:0dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:
dense/Tensordot/transpose	Transposeinputsdense/Tensordot/concat:output:0*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙dŹ
dense/Tensordot/ReshapeReshapedense/Tensordot/transpose:y:0dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
dense/Tensordot/MatMulMatMul dense/Tensordot/Reshape:output:0&dense/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Źb
dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Ź_
dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ż
dense/Tensordot/concat_1ConcatV2!dense/Tensordot/GatherV2:output:0 dense/Tensordot/Const_2:output:0&dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
dense/TensordotReshape dense/Tensordot/MatMul:product:0!dense/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙dŹ
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:Ź*
dtype0
dense/BiasAddBiasAdddense/Tensordot:output:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙dŹb
dense/LeakyRelu	LeakyReludense/BiasAdd:output:0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙dŹ
 dense_1/Tensordot/ReadVariableOpReadVariableOp)dense_1_tensordot_readvariableop_resource* 
_output_shapes
:
ŹŹ*
dtype0`
dense_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:g
dense_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       d
dense_1/Tensordot/ShapeShapedense/LeakyRelu:activations:0*
T0*
_output_shapes
:a
dense_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : Ű
dense_1/Tensordot/GatherV2GatherV2 dense_1/Tensordot/Shape:output:0dense_1/Tensordot/free:output:0(dense_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:c
!dense_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ß
dense_1/Tensordot/GatherV2_1GatherV2 dense_1/Tensordot/Shape:output:0dense_1/Tensordot/axes:output:0*dense_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
dense_1/Tensordot/ProdProd#dense_1/Tensordot/GatherV2:output:0 dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: c
dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
dense_1/Tensordot/Prod_1Prod%dense_1/Tensordot/GatherV2_1:output:0"dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ź
dense_1/Tensordot/concatConcatV2dense_1/Tensordot/free:output:0dense_1/Tensordot/axes:output:0&dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:
dense_1/Tensordot/stackPackdense_1/Tensordot/Prod:output:0!dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:Ą
dense_1/Tensordot/transpose	Transposedense/LeakyRelu:activations:0!dense_1/Tensordot/concat:output:0*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙dŹ˘
dense_1/Tensordot/ReshapeReshapedense_1/Tensordot/transpose:y:0 dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙Ł
dense_1/Tensordot/MatMulMatMul"dense_1/Tensordot/Reshape:output:0(dense_1/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Źd
dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Źa
dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ç
dense_1/Tensordot/concat_1ConcatV2#dense_1/Tensordot/GatherV2:output:0"dense_1/Tensordot/Const_2:output:0(dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
dense_1/TensordotReshape"dense_1/Tensordot/MatMul:product:0#dense_1/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙dŹ
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:Ź*
dtype0
dense_1/BiasAddBiasAdddense_1/Tensordot:output:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙dŹl
IdentityIdentitydense_1/BiasAdd:output:0^NoOp*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙dŹĘ
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/Tensordot/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp!^dense_1/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :˙˙˙˙˙˙˙˙˙dŹ: : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2@
dense/Tensordot/ReadVariableOpdense/Tensordot/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2D
 dense_1/Tensordot/ReadVariableOp dense_1/Tensordot/ReadVariableOp:T P
,
_output_shapes
:˙˙˙˙˙˙˙˙˙dŹ
 
_user_specified_nameinputs
ą
Í
I__inference_sequential_2_layer_call_and_return_conditional_losses_3832454

inputs"
dense_4_3832448:	Ź
dense_4_3832450:
identity˘dense_4/StatefulPartitionedCallď
dense_4/StatefulPartitionedCallStatefulPartitionedCallinputsdense_4_3832448dense_4_3832450*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_4_layer_call_and_return_conditional_losses_3832410w
IdentityIdentity(dense_4/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙h
NoOpNoOp ^dense_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:˙˙˙˙˙˙˙˙˙Ź: : 2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ź
 
_user_specified_nameinputs

Ž
G__inference_sequential_layer_call_and_return_conditional_losses_3832134

inputs!
dense_3832123:
ŹŹ
dense_3832125:	Ź#
dense_1_3832128:
ŹŹ
dense_1_3832130:	Ź
identity˘dense/StatefulPartitionedCall˘dense_1/StatefulPartitionedCallě
dense/StatefulPartitionedCallStatefulPartitionedCallinputsdense_3832123dense_3832125*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:˙˙˙˙˙˙˙˙˙dŹ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_3832031
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_3832128dense_1_3832130*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:˙˙˙˙˙˙˙˙˙dŹ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_3832067|
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙dŹ
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :˙˙˙˙˙˙˙˙˙dŹ: : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:T P
,
_output_shapes
:˙˙˙˙˙˙˙˙˙dŹ
 
_user_specified_nameinputs
Ě
­
>__inference_token_and_position_embedding_layer_call_fn_3834064
x
unknown:	dŹ
	unknown_0:
XŹ
identity˘StatefulPartitionedCallá
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	dŹ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *b
f]R[
Y__inference_token_and_position_embedding_layer_call_and_return_conditional_losses_3832522g
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes
:	dŹ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:˙˙˙˙˙˙˙˙˙: : 22
StatefulPartitionedCallStatefulPartitionedCall:F B
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_namex"
L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*
serving_default
7
input_1,
serving_default_input_1:0˙˙˙˙˙˙˙˙˙3
output_1'
StatefulPartitionedCall:0tensorflow/serving/predict:ř
Ť
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
embedding_layer
	enc_positional_embedding

enc_transformerblock
word2idx
accuracy_list
dec_transformer_block
dec_positional_embedding
dec_pooling
dec_classifier
dropout

signatures"
_tf_keras_model
ö
0
1
2
3
4
5
6
7
8
9
10
11
12
 13
!14
"15
#16
$17
%18
&19
'20
(21
)22
*23
+24
,25
-26
.27"
trackable_list_wrapper
ö
0
1
2
3
4
5
6
7
8
9
10
11
12
 13
!14
"15
#16
$17
%18
&19
'20
(21
)22
*23
+24
,25
-26
.27"
trackable_list_wrapper
 "
trackable_list_wrapper
Ę
/non_trainable_variables

0layers
1metrics
2layer_regularization_losses
3layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object

4trace_0
5trace_1
6trace_2
7trace_32Ź
9__inference_emotion_detection_model_layer_call_fn_3832929
9__inference_emotion_detection_model_layer_call_fn_3833827
9__inference_emotion_detection_model_layer_call_fn_3833882
9__inference_emotion_detection_model_layer_call_fn_3833591˝
´˛°
FullArgSpec
args
jself
jx_in
varargs
 
varkw
 
defaults
 

kwonlyargs

jtraining%
kwonlydefaultsŞ

trainingp 
annotationsŞ *
 z4trace_0z5trace_1z6trace_2z7trace_3

8trace_0
9trace_1
:trace_2
;trace_32
T__inference_emotion_detection_model_layer_call_and_return_conditional_losses_3833957
T__inference_emotion_detection_model_layer_call_and_return_conditional_losses_3834039
T__inference_emotion_detection_model_layer_call_and_return_conditional_losses_3833653
T__inference_emotion_detection_model_layer_call_and_return_conditional_losses_3833715˝
´˛°
FullArgSpec
args
jself
jx_in
varargs
 
varkw
 
defaults
 

kwonlyargs

jtraining%
kwonlydefaultsŞ

trainingp 
annotationsŞ *
 z8trace_0z9trace_1z:trace_2z;trace_3
ÍBĘ
"__inference__wrapped_model_3831993input_1"
˛
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
ľ
<	variables
=trainable_variables
>regularization_losses
?	keras_api
@__call__
*A&call_and_return_all_conditional_losses

embeddings"
_tf_keras_layer
Á
B	variables
Ctrainable_variables
Dregularization_losses
E	keras_api
F__call__
*G&call_and_return_all_conditional_losses
	token_emb
Hpos_emb"
_tf_keras_layer
ő
I	variables
Jtrainable_variables
Kregularization_losses
L	keras_api
M__call__
*N&call_and_return_all_conditional_losses
Off_layer
P
self_atten
Qself_context_atten
R
layer_norm
Scall"
_tf_keras_layer
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
ő
T	variables
Utrainable_variables
Vregularization_losses
W	keras_api
X__call__
*Y&call_and_return_all_conditional_losses
Zff_layer
[
self_atten
\self_context_atten
]
layer_norm
^call"
_tf_keras_layer
D
_	keras_api
	token_emb
`pos_emb"
_tf_keras_layer
Ľ
a	variables
btrainable_variables
cregularization_losses
d	keras_api
e__call__
*f&call_and_return_all_conditional_losses"
_tf_keras_layer
Ń
glayer_with_weights-0
glayer-0
h	variables
itrainable_variables
jregularization_losses
k	keras_api
l__call__
*m&call_and_return_all_conditional_losses"
_tf_keras_sequential
ź
n	variables
otrainable_variables
pregularization_losses
q	keras_api
r__call__
*s&call_and_return_all_conditional_losses
t_random_generator"
_tf_keras_layer
,
userving_default"
signature_map
]:[
XŹ2Iemotion_detection_model/token_and_position_embedding/embedding/embeddings
^:\	dŹ2Kemotion_detection_model/token_and_position_embedding/embedding_1/embeddings
 :
ŹŹ2dense/kernel
:Ź2
dense/bias
": 
ŹŹ2dense_1/kernel
:Ź2dense_1/bias
:
ŹŹ2Variable
:
ŹŹ2Variable
:
ŹŹ2Variable
:
ŹŹ2Variable
:
ŹŹ2Variable
:
ŹŹ2Variable
(:&Ź2layer_normalization/gamma
':%Ź2layer_normalization/beta
": 
ŹŹ2dense_2/kernel
:Ź2dense_2/bias
": 
ŹŹ2dense_3/kernel
:Ź2dense_3/bias
:
ŹŹ2Variable
:
ŹŹ2Variable
:
ŹŹ2Variable
:
ŹŹ2Variable
:
ŹŹ2Variable
:
ŹŹ2Variable
*:(Ź2layer_normalization_1/gamma
):'Ź2layer_normalization_1/beta
!:	Ź2dense_4/kernel
:2dense_4/bias
 "
trackable_list_wrapper
X
0
	1

2
3
4
5
6
7"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
B
9__inference_emotion_detection_model_layer_call_fn_3832929input_1"˝
´˛°
FullArgSpec
args
jself
jx_in
varargs
 
varkw
 
defaults
 

kwonlyargs

jtraining%
kwonlydefaultsŞ

trainingp 
annotationsŞ *
 
B
9__inference_emotion_detection_model_layer_call_fn_3833827x_in"˝
´˛°
FullArgSpec
args
jself
jx_in
varargs
 
varkw
 
defaults
 

kwonlyargs

jtraining%
kwonlydefaultsŞ

trainingp 
annotationsŞ *
 
B
9__inference_emotion_detection_model_layer_call_fn_3833882x_in"˝
´˛°
FullArgSpec
args
jself
jx_in
varargs
 
varkw
 
defaults
 

kwonlyargs

jtraining%
kwonlydefaultsŞ

trainingp 
annotationsŞ *
 
B
9__inference_emotion_detection_model_layer_call_fn_3833591input_1"˝
´˛°
FullArgSpec
args
jself
jx_in
varargs
 
varkw
 
defaults
 

kwonlyargs

jtraining%
kwonlydefaultsŞ

trainingp 
annotationsŞ *
 
ĄB
T__inference_emotion_detection_model_layer_call_and_return_conditional_losses_3833957x_in"˝
´˛°
FullArgSpec
args
jself
jx_in
varargs
 
varkw
 
defaults
 

kwonlyargs

jtraining%
kwonlydefaultsŞ

trainingp 
annotationsŞ *
 
ĄB
T__inference_emotion_detection_model_layer_call_and_return_conditional_losses_3834039x_in"˝
´˛°
FullArgSpec
args
jself
jx_in
varargs
 
varkw
 
defaults
 

kwonlyargs

jtraining%
kwonlydefaultsŞ

trainingp 
annotationsŞ *
 
¤BĄ
T__inference_emotion_detection_model_layer_call_and_return_conditional_losses_3833653input_1"˝
´˛°
FullArgSpec
args
jself
jx_in
varargs
 
varkw
 
defaults
 

kwonlyargs

jtraining%
kwonlydefaultsŞ

trainingp 
annotationsŞ *
 
¤BĄ
T__inference_emotion_detection_model_layer_call_and_return_conditional_losses_3833715input_1"˝
´˛°
FullArgSpec
args
jself
jx_in
varargs
 
varkw
 
defaults
 

kwonlyargs

jtraining%
kwonlydefaultsŞ

trainingp 
annotationsŞ *
 
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
­
vnon_trainable_variables

wlayers
xmetrics
ylayer_regularization_losses
zlayer_metrics
<	variables
=trainable_variables
>regularization_losses
@__call__
*A&call_and_return_all_conditional_losses
&A"call_and_return_conditional_losses"
_generic_user_object
ď
{trace_02Ň
+__inference_embedding_layer_call_fn_3834046˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 z{trace_0

|trace_02í
F__inference_embedding_layer_call_and_return_conditional_losses_3834055˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 z|trace_0
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
Ż
}non_trainable_variables

~layers
metrics
 layer_regularization_losses
layer_metrics
B	variables
Ctrainable_variables
Dregularization_losses
F__call__
*G&call_and_return_all_conditional_losses
&G"call_and_return_conditional_losses"
_generic_user_object
˙
trace_02ŕ
>__inference_token_and_position_embedding_layer_call_fn_3834064
˛
FullArgSpec
args
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 ztrace_0

trace_02ű
Y__inference_token_and_position_embedding_layer_call_and_return_conditional_losses_3834083
˛
FullArgSpec
args
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 ztrace_0
ť
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses

embeddings"
_tf_keras_layer
v
0
1
2
3
4
5
6
7
8
9
10
 11"
trackable_list_wrapper
v
0
1
2
3
4
5
6
7
8
9
10
 11"
trackable_list_wrapper
 "
trackable_list_wrapper
˛
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
I	variables
Jtrainable_variables
Kregularization_losses
M__call__
*N&call_and_return_all_conditional_losses
&N"call_and_return_conditional_losses"
_generic_user_object

trace_0
trace_12×
3__inference_transformer_block_layer_call_fn_3834106
3__inference_transformer_block_layer_call_fn_3834129ę
á˛Ý
FullArgSpec?
args74
jself
jinputs
jcontext_sequence
j
is_decoder
varargs
 
varkw
 
defaults˘

 
p 

kwonlyargs

jtraining%
kwonlydefaultsŞ

trainingp 
annotationsŞ *
 ztrace_0ztrace_1
Č
trace_0
trace_12
N__inference_transformer_block_layer_call_and_return_conditional_losses_3834247
N__inference_transformer_block_layer_call_and_return_conditional_losses_3834365ę
á˛Ý
FullArgSpec?
args74
jself
jinputs
jcontext_sequence
j
is_decoder
varargs
 
varkw
 
defaults˘

 
p 

kwonlyargs

jtraining%
kwonlydefaultsŞ

trainingp 
annotationsŞ *
 ztrace_0ztrace_1

layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_sequential
â
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+ &call_and_return_all_conditional_losses
K
V
Q
Ąattention_matrix
	˘call"
_tf_keras_layer
`
Ł	keras_api
K
V
Q
¤attention_matrix
	Ľcall"
_tf_keras_layer
Ë
Ś	variables
§trainable_variables
¨regularization_losses
Š	keras_api
Ş__call__
+Ť&call_and_return_all_conditional_losses
	Źaxis
	gamma
 beta"
_tf_keras_layer

­trace_02ę
__inference_call_3830033Í
Ä˛Ŕ
FullArgSpec?
args74
jself
jinputs
jcontext_sequence
j
is_decoder
varargs
 
varkw
 
defaults˘

 
p 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 z­trace_0
v
!0
"1
#2
$3
%4
&5
'6
(7
)8
*9
+10
,11"
trackable_list_wrapper
v
!0
"1
#2
$3
%4
&5
'6
(7
)8
*9
+10
,11"
trackable_list_wrapper
 "
trackable_list_wrapper
˛
Žnon_trainable_variables
Żlayers
°metrics
 ąlayer_regularization_losses
˛layer_metrics
T	variables
Utrainable_variables
Vregularization_losses
X__call__
*Y&call_and_return_all_conditional_losses
&Y"call_and_return_conditional_losses"
_generic_user_object

łtrace_0
´trace_12Ű
5__inference_transformer_block_1_layer_call_fn_3834395
5__inference_transformer_block_1_layer_call_fn_3834425ę
á˛Ý
FullArgSpec?
args74
jself
jinputs
jcontext_sequence
j
is_decoder
varargs
 
varkw
 
defaults˘

 
p 

kwonlyargs

jtraining%
kwonlydefaultsŞ

trainingp 
annotationsŞ *
 złtrace_0z´trace_1
Ě
ľtrace_0
śtrace_12
P__inference_transformer_block_1_layer_call_and_return_conditional_losses_3834593
P__inference_transformer_block_1_layer_call_and_return_conditional_losses_3834761ę
á˛Ý
FullArgSpec?
args74
jself
jinputs
jcontext_sequence
j
is_decoder
varargs
 
varkw
 
defaults˘

 
p 

kwonlyargs

jtraining%
kwonlydefaultsŞ

trainingp 
annotationsŞ *
 zľtrace_0zśtrace_1

ˇlayer_with_weights-0
ˇlayer-0
¸layer_with_weights-1
¸layer-1
š	variables
ştrainable_variables
ťregularization_losses
ź	keras_api
˝__call__
+ž&call_and_return_all_conditional_losses"
_tf_keras_sequential
â
ż	variables
Ŕtrainable_variables
Áregularization_losses
Â	keras_api
Ă__call__
+Ä&call_and_return_all_conditional_losses
%K
&V
'Q
Ĺattention_matrix
	Ćcall"
_tf_keras_layer
â
Ç	variables
Čtrainable_variables
Éregularization_losses
Ę	keras_api
Ë__call__
+Ě&call_and_return_all_conditional_losses
(K
)V
*Q
Íattention_matrix
	Îcall"
_tf_keras_layer
Ë
Ď	variables
Đtrainable_variables
Ńregularization_losses
Ň	keras_api
Ó__call__
+Ô&call_and_return_all_conditional_losses
	Őaxis
	+gamma
,beta"
_tf_keras_layer

Ötrace_02ę
__inference_call_3830201Í
Ä˛Ŕ
FullArgSpec?
args74
jself
jinputs
jcontext_sequence
j
is_decoder
varargs
 
varkw
 
defaults˘

 
p 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 zÖtrace_0
"
_generic_user_object
)
×	keras_api"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
˛
Řnon_trainable_variables
Ůlayers
Úmetrics
 Űlayer_regularization_losses
Ülayer_metrics
a	variables
btrainable_variables
cregularization_losses
e__call__
*f&call_and_return_all_conditional_losses
&f"call_and_return_conditional_losses"
_generic_user_object
ü
Ýtrace_02Ý
6__inference_global_max_pooling2d_layer_call_fn_3834766˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 zÝtrace_0

Ţtrace_02ř
Q__inference_global_max_pooling2d_layer_call_and_return_conditional_losses_3834772˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 zŢtrace_0
Á
ß	variables
ŕtrainable_variables
áregularization_losses
â	keras_api
ă__call__
+ä&call_and_return_all_conditional_losses

-kernel
.bias"
_tf_keras_layer
.
-0
.1"
trackable_list_wrapper
.
-0
.1"
trackable_list_wrapper
 "
trackable_list_wrapper
˛
ĺnon_trainable_variables
ćlayers
çmetrics
 člayer_regularization_losses
élayer_metrics
h	variables
itrainable_variables
jregularization_losses
l__call__
*m&call_and_return_all_conditional_losses
&m"call_and_return_conditional_losses"
_generic_user_object
ő
ętrace_0
ëtrace_1
ětrace_2
ítrace_32
.__inference_sequential_2_layer_call_fn_3832424
.__inference_sequential_2_layer_call_fn_3834781
.__inference_sequential_2_layer_call_fn_3834790
.__inference_sequential_2_layer_call_fn_3832470ż
ś˛˛
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 zętrace_0zëtrace_1zětrace_2zítrace_3
á
îtrace_0
ďtrace_1
đtrace_2
ńtrace_32î
I__inference_sequential_2_layer_call_and_return_conditional_losses_3834801
I__inference_sequential_2_layer_call_and_return_conditional_losses_3834812
I__inference_sequential_2_layer_call_and_return_conditional_losses_3832479
I__inference_sequential_2_layer_call_and_return_conditional_losses_3832488ż
ś˛˛
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 zîtrace_0zďtrace_1zđtrace_2zńtrace_3
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
˛
ňnon_trainable_variables
ólayers
ômetrics
 őlayer_regularization_losses
ölayer_metrics
n	variables
otrainable_variables
pregularization_losses
r__call__
*s&call_and_return_all_conditional_losses
&s"call_and_return_conditional_losses"
_generic_user_object
Ç
÷trace_0
řtrace_12
)__inference_dropout_layer_call_fn_3834817
)__inference_dropout_layer_call_fn_3834822ł
Ş˛Ś
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 z÷trace_0zřtrace_1
ý
ůtrace_0
útrace_12Â
D__inference_dropout_layer_call_and_return_conditional_losses_3834827
D__inference_dropout_layer_call_and_return_conditional_losses_3834839ł
Ş˛Ś
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 zůtrace_0zútrace_1
"
_generic_user_object
ĚBÉ
%__inference_signature_wrapper_3833772input_1"
˛
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ßBÜ
+__inference_embedding_layer_call_fn_3834046inputs"˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
úB÷
F__inference_embedding_layer_call_and_return_conditional_losses_3834055inputs"˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
 "
trackable_list_wrapper
.
0
H1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
čBĺ
>__inference_token_and_position_embedding_layer_call_fn_3834064x"
˛
FullArgSpec
args
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
B
Y__inference_token_and_position_embedding_layer_call_and_return_conditional_losses_3834083x"
˛
FullArgSpec
args
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
űnon_trainable_variables
ülayers
ýmetrics
 ţlayer_regularization_losses
˙layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
¨2Ľ˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
¨2Ľ˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
 "
trackable_list_wrapper
<
O0
P1
Q2
R3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ŻBŹ
3__inference_transformer_block_layer_call_fn_3834106inputs"ę
á˛Ý
FullArgSpec?
args74
jself
jinputs
jcontext_sequence
j
is_decoder
varargs
 
varkw
 
defaults˘

 
p 

kwonlyargs

jtraining%
kwonlydefaultsŞ

trainingp 
annotationsŞ *
 
ŻBŹ
3__inference_transformer_block_layer_call_fn_3834129inputs"ę
á˛Ý
FullArgSpec?
args74
jself
jinputs
jcontext_sequence
j
is_decoder
varargs
 
varkw
 
defaults˘

 
p 

kwonlyargs

jtraining%
kwonlydefaultsŞ

trainingp 
annotationsŞ *
 
ĘBÇ
N__inference_transformer_block_layer_call_and_return_conditional_losses_3834247inputs"ę
á˛Ý
FullArgSpec?
args74
jself
jinputs
jcontext_sequence
j
is_decoder
varargs
 
varkw
 
defaults˘

 
p 

kwonlyargs

jtraining%
kwonlydefaultsŞ

trainingp 
annotationsŞ *
 
ĘBÇ
N__inference_transformer_block_layer_call_and_return_conditional_losses_3834365inputs"ę
á˛Ý
FullArgSpec?
args74
jself
jinputs
jcontext_sequence
j
is_decoder
varargs
 
varkw
 
defaults˘

 
p 

kwonlyargs

jtraining%
kwonlydefaultsŞ

trainingp 
annotationsŞ *
 
Á
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
Á
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
<
0
1
2
3"
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
í
trace_0
trace_1
trace_2
trace_32ú
,__inference_sequential_layer_call_fn_3832085
,__inference_sequential_layer_call_fn_3834852
,__inference_sequential_layer_call_fn_3834865
,__inference_sequential_layer_call_fn_3832158ż
ś˛˛
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 ztrace_0ztrace_1ztrace_2ztrace_3
Ů
trace_0
trace_1
trace_2
trace_32ć
G__inference_sequential_layer_call_and_return_conditional_losses_3834922
G__inference_sequential_layer_call_and_return_conditional_losses_3834979
G__inference_sequential_layer_call_and_return_conditional_losses_3832172
G__inference_sequential_layer_call_and_return_conditional_losses_3832186ż
ś˛˛
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 ztrace_0ztrace_1ztrace_2ztrace_3
5
0
1
2"
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+ &call_and_return_all_conditional_losses
' "call_and_return_conditional_losses"
_generic_user_object
Ü2ŮÖ
Í˛É
FullArgSpecQ
argsIF
jself
jinputs_for_keys
jinputs_for_values
jinputs_for_queries
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
Ü2ŮÖ
Í˛É
FullArgSpecQ
argsIF
jself
jinputs_for_keys
jinputs_for_values
jinputs_for_queries
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
Ť
	variables
trainable_variables
 regularization_losses
Ą	keras_api
˘__call__
+Ł&call_and_return_all_conditional_losses"
_tf_keras_layer

¤trace_02ó
__inference_call_3830564Ö
Í˛É
FullArgSpecQ
argsIF
jself
jinputs_for_keys
jinputs_for_values
jinputs_for_queries
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 z¤trace_0
"
_generic_user_object
)
Ľ	keras_api"
_tf_keras_layer
Ü2ŮÖ
Í˛É
FullArgSpecQ
argsIF
jself
jinputs_for_keys
jinputs_for_values
jinputs_for_queries
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
.
0
 1"
trackable_list_wrapper
.
0
 1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Śnon_trainable_variables
§layers
¨metrics
 Šlayer_regularization_losses
Şlayer_metrics
Ś	variables
§trainable_variables
¨regularization_losses
Ş__call__
+Ť&call_and_return_all_conditional_losses
'Ť"call_and_return_conditional_losses"
_generic_user_object
¨2Ľ˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
¨2Ľ˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
 "
trackable_list_wrapper
÷Bô
__inference_call_3830033inputs"Í
Ä˛Ŕ
FullArgSpec?
args74
jself
jinputs
jcontext_sequence
j
is_decoder
varargs
 
varkw
 
defaults˘

 
p 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
 "
trackable_list_wrapper
<
Z0
[1
\2
]3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ĂBŔ
5__inference_transformer_block_1_layer_call_fn_3834395inputscontext_sequence"ę
á˛Ý
FullArgSpec?
args74
jself
jinputs
jcontext_sequence
j
is_decoder
varargs
 
varkw
 
defaults˘

 
p 

kwonlyargs

jtraining%
kwonlydefaultsŞ

trainingp 
annotationsŞ *
 
ĂBŔ
5__inference_transformer_block_1_layer_call_fn_3834425inputscontext_sequence"ę
á˛Ý
FullArgSpec?
args74
jself
jinputs
jcontext_sequence
j
is_decoder
varargs
 
varkw
 
defaults˘

 
p 

kwonlyargs

jtraining%
kwonlydefaultsŞ

trainingp 
annotationsŞ *
 
ŢBŰ
P__inference_transformer_block_1_layer_call_and_return_conditional_losses_3834593inputscontext_sequence"ę
á˛Ý
FullArgSpec?
args74
jself
jinputs
jcontext_sequence
j
is_decoder
varargs
 
varkw
 
defaults˘

 
p 

kwonlyargs

jtraining%
kwonlydefaultsŞ

trainingp 
annotationsŞ *
 
ŢBŰ
P__inference_transformer_block_1_layer_call_and_return_conditional_losses_3834761inputscontext_sequence"ę
á˛Ý
FullArgSpec?
args74
jself
jinputs
jcontext_sequence
j
is_decoder
varargs
 
varkw
 
defaults˘

 
p 

kwonlyargs

jtraining%
kwonlydefaultsŞ

trainingp 
annotationsŞ *
 
Á
Ť	variables
Źtrainable_variables
­regularization_losses
Ž	keras_api
Ż__call__
+°&call_and_return_all_conditional_losses

!kernel
"bias"
_tf_keras_layer
Á
ą	variables
˛trainable_variables
łregularization_losses
´	keras_api
ľ__call__
+ś&call_and_return_all_conditional_losses

#kernel
$bias"
_tf_keras_layer
<
!0
"1
#2
$3"
trackable_list_wrapper
<
!0
"1
#2
$3"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ˇnon_trainable_variables
¸layers
šmetrics
 şlayer_regularization_losses
ťlayer_metrics
š	variables
ştrainable_variables
ťregularization_losses
˝__call__
+ž&call_and_return_all_conditional_losses
'ž"call_and_return_conditional_losses"
_generic_user_object
ő
źtrace_0
˝trace_1
žtrace_2
żtrace_32
.__inference_sequential_1_layer_call_fn_3832278
.__inference_sequential_1_layer_call_fn_3834992
.__inference_sequential_1_layer_call_fn_3835005
.__inference_sequential_1_layer_call_fn_3832351ż
ś˛˛
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 zźtrace_0z˝trace_1zžtrace_2zżtrace_3
á
Ŕtrace_0
Átrace_1
Âtrace_2
Ătrace_32î
I__inference_sequential_1_layer_call_and_return_conditional_losses_3835062
I__inference_sequential_1_layer_call_and_return_conditional_losses_3835119
I__inference_sequential_1_layer_call_and_return_conditional_losses_3832365
I__inference_sequential_1_layer_call_and_return_conditional_losses_3832379ż
ś˛˛
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 zŔtrace_0zÁtrace_1zÂtrace_2zĂtrace_3
5
%0
&1
'2"
trackable_list_wrapper
5
%0
&1
'2"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Änon_trainable_variables
Ĺlayers
Ćmetrics
 Çlayer_regularization_losses
Člayer_metrics
ż	variables
Ŕtrainable_variables
Áregularization_losses
Ă__call__
+Ä&call_and_return_all_conditional_losses
'Ä"call_and_return_conditional_losses"
_generic_user_object
Ü2ŮÖ
Í˛É
FullArgSpecQ
argsIF
jself
jinputs_for_keys
jinputs_for_values
jinputs_for_queries
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
Ü2ŮÖ
Í˛É
FullArgSpecQ
argsIF
jself
jinputs_for_keys
jinputs_for_values
jinputs_for_queries
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
Ť
É	variables
Ętrainable_variables
Ëregularization_losses
Ě	keras_api
Í__call__
+Î&call_and_return_all_conditional_losses"
_tf_keras_layer

Ďtrace_02ó
__inference_call_3831009Ö
Í˛É
FullArgSpecQ
argsIF
jself
jinputs_for_keys
jinputs_for_values
jinputs_for_queries
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 zĎtrace_0
5
(0
)1
*2"
trackable_list_wrapper
5
(0
)1
*2"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Đnon_trainable_variables
Ńlayers
Ňmetrics
 Ólayer_regularization_losses
Ôlayer_metrics
Ç	variables
Čtrainable_variables
Éregularization_losses
Ë__call__
+Ě&call_and_return_all_conditional_losses
'Ě"call_and_return_conditional_losses"
_generic_user_object
Ü2ŮÖ
Í˛É
FullArgSpecQ
argsIF
jself
jinputs_for_keys
jinputs_for_values
jinputs_for_queries
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
Ü2ŮÖ
Í˛É
FullArgSpecQ
argsIF
jself
jinputs_for_keys
jinputs_for_values
jinputs_for_queries
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
Ť
Ő	variables
Ötrainable_variables
×regularization_losses
Ř	keras_api
Ů__call__
+Ú&call_and_return_all_conditional_losses"
_tf_keras_layer

Űtrace_02ó
__inference_call_3831053Ö
Í˛É
FullArgSpecQ
argsIF
jself
jinputs_for_keys
jinputs_for_values
jinputs_for_queries
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 zŰtrace_0
.
+0
,1"
trackable_list_wrapper
.
+0
,1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Ünon_trainable_variables
Ýlayers
Ţmetrics
 ßlayer_regularization_losses
ŕlayer_metrics
Ď	variables
Đtrainable_variables
Ńregularization_losses
Ó__call__
+Ô&call_and_return_all_conditional_losses
'Ô"call_and_return_conditional_losses"
_generic_user_object
¨2Ľ˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
¨2Ľ˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
 "
trackable_list_wrapper
B
__inference_call_3830201inputscontext_sequence"Í
Ä˛Ŕ
FullArgSpec?
args74
jself
jinputs
jcontext_sequence
j
is_decoder
varargs
 
varkw
 
defaults˘

 
p 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ęBç
6__inference_global_max_pooling2d_layer_call_fn_3834766inputs"˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
B
Q__inference_global_max_pooling2d_layer_call_and_return_conditional_losses_3834772inputs"˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
.
-0
.1"
trackable_list_wrapper
.
-0
.1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ánon_trainable_variables
âlayers
ămetrics
 älayer_regularization_losses
ĺlayer_metrics
ß	variables
ŕtrainable_variables
áregularization_losses
ă__call__
+ä&call_and_return_all_conditional_losses
'ä"call_and_return_conditional_losses"
_generic_user_object
ď
ćtrace_02Đ
)__inference_dense_4_layer_call_fn_3835128˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 zćtrace_0

çtrace_02ë
D__inference_dense_4_layer_call_and_return_conditional_losses_3835139˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 zçtrace_0
 "
trackable_list_wrapper
'
g0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
B
.__inference_sequential_2_layer_call_fn_3832424dense_4_input"ż
ś˛˛
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
˙Bü
.__inference_sequential_2_layer_call_fn_3834781inputs"ż
ś˛˛
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
˙Bü
.__inference_sequential_2_layer_call_fn_3834790inputs"ż
ś˛˛
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
B
.__inference_sequential_2_layer_call_fn_3832470dense_4_input"ż
ś˛˛
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
B
I__inference_sequential_2_layer_call_and_return_conditional_losses_3834801inputs"ż
ś˛˛
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
B
I__inference_sequential_2_layer_call_and_return_conditional_losses_3834812inputs"ż
ś˛˛
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
ĄB
I__inference_sequential_2_layer_call_and_return_conditional_losses_3832479dense_4_input"ż
ś˛˛
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
ĄB
I__inference_sequential_2_layer_call_and_return_conditional_losses_3832488dense_4_input"ż
ś˛˛
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
îBë
)__inference_dropout_layer_call_fn_3834817inputs"ł
Ş˛Ś
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
îBë
)__inference_dropout_layer_call_fn_3834822inputs"ł
Ş˛Ś
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
B
D__inference_dropout_layer_call_and_return_conditional_losses_3834827inputs"ł
Ş˛Ś
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
B
D__inference_dropout_layer_call_and_return_conditional_losses_3834839inputs"ł
Ş˛Ś
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
čnon_trainable_variables
élayers
ęmetrics
 ëlayer_regularization_losses
ělayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
í
ítrace_02Î
'__inference_dense_layer_call_fn_3835148˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 zítrace_0

îtrace_02é
B__inference_dense_layer_call_and_return_conditional_losses_3835179˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 zîtrace_0
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ďnon_trainable_variables
đlayers
ńmetrics
 ňlayer_regularization_losses
ólayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
ď
ôtrace_02Đ
)__inference_dense_1_layer_call_fn_3835188˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 zôtrace_0

őtrace_02ë
D__inference_dense_1_layer_call_and_return_conditional_losses_3835218˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 zőtrace_0
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
B˙
,__inference_sequential_layer_call_fn_3832085dense_input"ż
ś˛˛
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
ýBú
,__inference_sequential_layer_call_fn_3834852inputs"ż
ś˛˛
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
ýBú
,__inference_sequential_layer_call_fn_3834865inputs"ż
ś˛˛
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
B˙
,__inference_sequential_layer_call_fn_3832158dense_input"ż
ś˛˛
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
B
G__inference_sequential_layer_call_and_return_conditional_losses_3834922inputs"ż
ś˛˛
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
B
G__inference_sequential_layer_call_and_return_conditional_losses_3834979inputs"ż
ś˛˛
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
B
G__inference_sequential_layer_call_and_return_conditional_losses_3832172dense_input"ż
ś˛˛
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
B
G__inference_sequential_layer_call_and_return_conditional_losses_3832186dense_input"ż
ś˛˛
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
 "
trackable_list_wrapper
(
Ą0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
önon_trainable_variables
÷layers
řmetrics
 ůlayer_regularization_losses
úlayer_metrics
	variables
trainable_variables
 regularization_losses
˘__call__
+Ł&call_and_return_all_conditional_losses
'Ł"call_and_return_conditional_losses"
_generic_user_object
¨2Ľ˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
¨2Ľ˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
°B­
__inference_call_3830564inputs_for_keysinputs_for_valuesinputs_for_queries"Ö
Í˛É
FullArgSpecQ
argsIF
jself
jinputs_for_keys
jinputs_for_values
jinputs_for_queries
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
!0
"1"
trackable_list_wrapper
.
!0
"1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
űnon_trainable_variables
ülayers
ýmetrics
 ţlayer_regularization_losses
˙layer_metrics
Ť	variables
Źtrainable_variables
­regularization_losses
Ż__call__
+°&call_and_return_all_conditional_losses
'°"call_and_return_conditional_losses"
_generic_user_object
ď
trace_02Đ
)__inference_dense_2_layer_call_fn_3835227˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 ztrace_0

trace_02ë
D__inference_dense_2_layer_call_and_return_conditional_losses_3835258˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 ztrace_0
.
#0
$1"
trackable_list_wrapper
.
#0
$1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
ą	variables
˛trainable_variables
łregularization_losses
ľ__call__
+ś&call_and_return_all_conditional_losses
'ś"call_and_return_conditional_losses"
_generic_user_object
ď
trace_02Đ
)__inference_dense_3_layer_call_fn_3835267˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 ztrace_0

trace_02ë
D__inference_dense_3_layer_call_and_return_conditional_losses_3835297˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 ztrace_0
 "
trackable_list_wrapper
0
ˇ0
¸1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
B
.__inference_sequential_1_layer_call_fn_3832278dense_2_input"ż
ś˛˛
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
˙Bü
.__inference_sequential_1_layer_call_fn_3834992inputs"ż
ś˛˛
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
˙Bü
.__inference_sequential_1_layer_call_fn_3835005inputs"ż
ś˛˛
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
B
.__inference_sequential_1_layer_call_fn_3832351dense_2_input"ż
ś˛˛
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
B
I__inference_sequential_1_layer_call_and_return_conditional_losses_3835062inputs"ż
ś˛˛
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
B
I__inference_sequential_1_layer_call_and_return_conditional_losses_3835119inputs"ż
ś˛˛
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
ĄB
I__inference_sequential_1_layer_call_and_return_conditional_losses_3832365dense_2_input"ż
ś˛˛
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
ĄB
I__inference_sequential_1_layer_call_and_return_conditional_losses_3832379dense_2_input"ż
ś˛˛
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
 "
trackable_list_wrapper
(
Ĺ0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
É	variables
Ętrainable_variables
Ëregularization_losses
Í__call__
+Î&call_and_return_all_conditional_losses
'Î"call_and_return_conditional_losses"
_generic_user_object
¨2Ľ˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
¨2Ľ˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
°B­
__inference_call_3831009inputs_for_keysinputs_for_valuesinputs_for_queries"Ö
Í˛É
FullArgSpecQ
argsIF
jself
jinputs_for_keys
jinputs_for_values
jinputs_for_queries
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
 "
trackable_list_wrapper
(
Í0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
Ő	variables
Ötrainable_variables
×regularization_losses
Ů__call__
+Ú&call_and_return_all_conditional_losses
'Ú"call_and_return_conditional_losses"
_generic_user_object
¨2Ľ˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
¨2Ľ˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
°B­
__inference_call_3831053inputs_for_keysinputs_for_valuesinputs_for_queries"Ö
Í˛É
FullArgSpecQ
argsIF
jself
jinputs_for_keys
jinputs_for_values
jinputs_for_queries
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ÝBÚ
)__inference_dense_4_layer_call_fn_3835128inputs"˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
řBő
D__inference_dense_4_layer_call_and_return_conditional_losses_3835139inputs"˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ŰBŘ
'__inference_dense_layer_call_fn_3835148inputs"˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
öBó
B__inference_dense_layer_call_and_return_conditional_losses_3835179inputs"˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ÝBÚ
)__inference_dense_1_layer_call_fn_3835188inputs"˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
řBő
D__inference_dense_1_layer_call_and_return_conditional_losses_3835218inputs"˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ÝBÚ
)__inference_dense_2_layer_call_fn_3835227inputs"˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
řBő
D__inference_dense_2_layer_call_and_return_conditional_losses_3835258inputs"˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ÝBÚ
)__inference_dense_3_layer_call_fn_3835267inputs"˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
řBő
D__inference_dense_3_layer_call_and_return_conditional_losses_3835297inputs"˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
"__inference__wrapped_model_3831993u %&'+,()*!"#$-.,˘)
"˘

input_1˙˙˙˙˙˙˙˙˙
Ş "*Ş'
%
output_1
output_1w
__inference_call_3830033[	 /˘,
%˘"

inputs	dŹ

 
p 
Ş "
unknownddŹ
__inference_call_3830201%&'+,()*!"#$S˘P
I˘F

inputsddŹ
"
context_sequence	dŹ
p
Ş "
unknownddŹž
__inference_call_3830564Ą{˘x
q˘n
!
inputs_for_keys	dŹ
# 
inputs_for_values	dŹ
$!
inputs_for_queries	dŹ
Ş "
unknownddŹĚ
__inference_call_3831009Ż%&'˘
}˘z
%"
inputs_for_keysddŹ
'$
inputs_for_valuesddŹ
(%
inputs_for_queriesddŹ
Ş "
unknownddŹČ
__inference_call_3831053Ť()*˘
y˘v
%"
inputs_for_keysddŹ
'$
inputs_for_valuesddŹ
$!
inputs_for_queries	dŹ
Ş "
unknownddŹľ
D__inference_dense_1_layer_call_and_return_conditional_losses_3835218m4˘1
*˘'
%"
inputs˙˙˙˙˙˙˙˙˙dŹ
Ş "1˘.
'$
tensor_0˙˙˙˙˙˙˙˙˙dŹ
 
)__inference_dense_1_layer_call_fn_3835188b4˘1
*˘'
%"
inputs˙˙˙˙˙˙˙˙˙dŹ
Ş "&#
unknown˙˙˙˙˙˙˙˙˙dŹľ
D__inference_dense_2_layer_call_and_return_conditional_losses_3835258m!"4˘1
*˘'
%"
inputs˙˙˙˙˙˙˙˙˙dŹ
Ş "1˘.
'$
tensor_0˙˙˙˙˙˙˙˙˙dŹ
 
)__inference_dense_2_layer_call_fn_3835227b!"4˘1
*˘'
%"
inputs˙˙˙˙˙˙˙˙˙dŹ
Ş "&#
unknown˙˙˙˙˙˙˙˙˙dŹľ
D__inference_dense_3_layer_call_and_return_conditional_losses_3835297m#$4˘1
*˘'
%"
inputs˙˙˙˙˙˙˙˙˙dŹ
Ş "1˘.
'$
tensor_0˙˙˙˙˙˙˙˙˙dŹ
 
)__inference_dense_3_layer_call_fn_3835267b#$4˘1
*˘'
%"
inputs˙˙˙˙˙˙˙˙˙dŹ
Ş "&#
unknown˙˙˙˙˙˙˙˙˙dŹŹ
D__inference_dense_4_layer_call_and_return_conditional_losses_3835139d-.0˘-
&˘#
!
inputs˙˙˙˙˙˙˙˙˙Ź
Ş ",˘)
"
tensor_0˙˙˙˙˙˙˙˙˙
 
)__inference_dense_4_layer_call_fn_3835128Y-.0˘-
&˘#
!
inputs˙˙˙˙˙˙˙˙˙Ź
Ş "!
unknown˙˙˙˙˙˙˙˙˙ł
B__inference_dense_layer_call_and_return_conditional_losses_3835179m4˘1
*˘'
%"
inputs˙˙˙˙˙˙˙˙˙dŹ
Ş "1˘.
'$
tensor_0˙˙˙˙˙˙˙˙˙dŹ
 
'__inference_dense_layer_call_fn_3835148b4˘1
*˘'
%"
inputs˙˙˙˙˙˙˙˙˙dŹ
Ş "&#
unknown˙˙˙˙˙˙˙˙˙dŹŁ
D__inference_dropout_layer_call_and_return_conditional_losses_3834827[/˘,
%˘"

inputsddŹ
p 
Ş "(˘%

tensor_0ddŹ
 Ł
D__inference_dropout_layer_call_and_return_conditional_losses_3834839[/˘,
%˘"

inputsddŹ
p
Ş "(˘%

tensor_0ddŹ
 }
)__inference_dropout_layer_call_fn_3834817P/˘,
%˘"

inputsddŹ
p 
Ş "
unknownddŹ}
)__inference_dropout_layer_call_fn_3834822P/˘,
%˘"

inputsddŹ
p
Ş "
unknownddŹŠ
F__inference_embedding_layer_call_and_return_conditional_losses_3834055_+˘(
!˘

inputs˙˙˙˙˙˙˙˙˙
Ş "-˘*
# 
tensor_0˙˙˙˙˙˙˙˙˙Ź
 
+__inference_embedding_layer_call_fn_3834046T+˘(
!˘

inputs˙˙˙˙˙˙˙˙˙
Ş ""
unknown˙˙˙˙˙˙˙˙˙ŹÖ
T__inference_emotion_detection_model_layer_call_and_return_conditional_losses_3833653~ %&'+,()*!"#$-.<˘9
"˘

input_1˙˙˙˙˙˙˙˙˙
Ş

trainingp "#˘ 

tensor_0
 Ö
T__inference_emotion_detection_model_layer_call_and_return_conditional_losses_3833715~ %&'+,()*!"#$-.<˘9
"˘

input_1˙˙˙˙˙˙˙˙˙
Ş

trainingp"#˘ 

tensor_0
 Ó
T__inference_emotion_detection_model_layer_call_and_return_conditional_losses_3833957{ %&'+,()*!"#$-.9˘6
˘

x_in˙˙˙˙˙˙˙˙˙
Ş

trainingp "#˘ 

tensor_0
 Ó
T__inference_emotion_detection_model_layer_call_and_return_conditional_losses_3834039{ %&'+,()*!"#$-.9˘6
˘

x_in˙˙˙˙˙˙˙˙˙
Ş

trainingp"#˘ 

tensor_0
 °
9__inference_emotion_detection_model_layer_call_fn_3832929s %&'+,()*!"#$-.<˘9
"˘

input_1˙˙˙˙˙˙˙˙˙
Ş

trainingp "
unknown°
9__inference_emotion_detection_model_layer_call_fn_3833591s %&'+,()*!"#$-.<˘9
"˘

input_1˙˙˙˙˙˙˙˙˙
Ş

trainingp"
unknown­
9__inference_emotion_detection_model_layer_call_fn_3833827p %&'+,()*!"#$-.9˘6
˘

x_in˙˙˙˙˙˙˙˙˙
Ş

trainingp "
unknown­
9__inference_emotion_detection_model_layer_call_fn_3833882p %&'+,()*!"#$-.9˘6
˘

x_in˙˙˙˙˙˙˙˙˙
Ş

trainingp"
unknowná
Q__inference_global_max_pooling2d_layer_call_and_return_conditional_losses_3834772R˘O
H˘E
C@
inputs4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Ş "5˘2
+(
tensor_0˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 ť
6__inference_global_max_pooling2d_layer_call_fn_3834766R˘O
H˘E
C@
inputs4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Ş "*'
unknown˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙Ë
I__inference_sequential_1_layer_call_and_return_conditional_losses_3832365~!"#$C˘@
9˘6
,)
dense_2_input˙˙˙˙˙˙˙˙˙dŹ
p 

 
Ş "1˘.
'$
tensor_0˙˙˙˙˙˙˙˙˙dŹ
 Ë
I__inference_sequential_1_layer_call_and_return_conditional_losses_3832379~!"#$C˘@
9˘6
,)
dense_2_input˙˙˙˙˙˙˙˙˙dŹ
p

 
Ş "1˘.
'$
tensor_0˙˙˙˙˙˙˙˙˙dŹ
 Ä
I__inference_sequential_1_layer_call_and_return_conditional_losses_3835062w!"#$<˘9
2˘/
%"
inputs˙˙˙˙˙˙˙˙˙dŹ
p 

 
Ş "1˘.
'$
tensor_0˙˙˙˙˙˙˙˙˙dŹ
 Ä
I__inference_sequential_1_layer_call_and_return_conditional_losses_3835119w!"#$<˘9
2˘/
%"
inputs˙˙˙˙˙˙˙˙˙dŹ
p

 
Ş "1˘.
'$
tensor_0˙˙˙˙˙˙˙˙˙dŹ
 Ľ
.__inference_sequential_1_layer_call_fn_3832278s!"#$C˘@
9˘6
,)
dense_2_input˙˙˙˙˙˙˙˙˙dŹ
p 

 
Ş "&#
unknown˙˙˙˙˙˙˙˙˙dŹĽ
.__inference_sequential_1_layer_call_fn_3832351s!"#$C˘@
9˘6
,)
dense_2_input˙˙˙˙˙˙˙˙˙dŹ
p

 
Ş "&#
unknown˙˙˙˙˙˙˙˙˙dŹ
.__inference_sequential_1_layer_call_fn_3834992l!"#$<˘9
2˘/
%"
inputs˙˙˙˙˙˙˙˙˙dŹ
p 

 
Ş "&#
unknown˙˙˙˙˙˙˙˙˙dŹ
.__inference_sequential_1_layer_call_fn_3835005l!"#$<˘9
2˘/
%"
inputs˙˙˙˙˙˙˙˙˙dŹ
p

 
Ş "&#
unknown˙˙˙˙˙˙˙˙˙dŹŔ
I__inference_sequential_2_layer_call_and_return_conditional_losses_3832479s-.?˘<
5˘2
(%
dense_4_input˙˙˙˙˙˙˙˙˙Ź
p 

 
Ş ",˘)
"
tensor_0˙˙˙˙˙˙˙˙˙
 Ŕ
I__inference_sequential_2_layer_call_and_return_conditional_losses_3832488s-.?˘<
5˘2
(%
dense_4_input˙˙˙˙˙˙˙˙˙Ź
p

 
Ş ",˘)
"
tensor_0˙˙˙˙˙˙˙˙˙
 š
I__inference_sequential_2_layer_call_and_return_conditional_losses_3834801l-.8˘5
.˘+
!
inputs˙˙˙˙˙˙˙˙˙Ź
p 

 
Ş ",˘)
"
tensor_0˙˙˙˙˙˙˙˙˙
 š
I__inference_sequential_2_layer_call_and_return_conditional_losses_3834812l-.8˘5
.˘+
!
inputs˙˙˙˙˙˙˙˙˙Ź
p

 
Ş ",˘)
"
tensor_0˙˙˙˙˙˙˙˙˙
 
.__inference_sequential_2_layer_call_fn_3832424h-.?˘<
5˘2
(%
dense_4_input˙˙˙˙˙˙˙˙˙Ź
p 

 
Ş "!
unknown˙˙˙˙˙˙˙˙˙
.__inference_sequential_2_layer_call_fn_3832470h-.?˘<
5˘2
(%
dense_4_input˙˙˙˙˙˙˙˙˙Ź
p

 
Ş "!
unknown˙˙˙˙˙˙˙˙˙
.__inference_sequential_2_layer_call_fn_3834781a-.8˘5
.˘+
!
inputs˙˙˙˙˙˙˙˙˙Ź
p 

 
Ş "!
unknown˙˙˙˙˙˙˙˙˙
.__inference_sequential_2_layer_call_fn_3834790a-.8˘5
.˘+
!
inputs˙˙˙˙˙˙˙˙˙Ź
p

 
Ş "!
unknown˙˙˙˙˙˙˙˙˙Ç
G__inference_sequential_layer_call_and_return_conditional_losses_3832172|A˘>
7˘4
*'
dense_input˙˙˙˙˙˙˙˙˙dŹ
p 

 
Ş "1˘.
'$
tensor_0˙˙˙˙˙˙˙˙˙dŹ
 Ç
G__inference_sequential_layer_call_and_return_conditional_losses_3832186|A˘>
7˘4
*'
dense_input˙˙˙˙˙˙˙˙˙dŹ
p

 
Ş "1˘.
'$
tensor_0˙˙˙˙˙˙˙˙˙dŹ
 Â
G__inference_sequential_layer_call_and_return_conditional_losses_3834922w<˘9
2˘/
%"
inputs˙˙˙˙˙˙˙˙˙dŹ
p 

 
Ş "1˘.
'$
tensor_0˙˙˙˙˙˙˙˙˙dŹ
 Â
G__inference_sequential_layer_call_and_return_conditional_losses_3834979w<˘9
2˘/
%"
inputs˙˙˙˙˙˙˙˙˙dŹ
p

 
Ş "1˘.
'$
tensor_0˙˙˙˙˙˙˙˙˙dŹ
 Ą
,__inference_sequential_layer_call_fn_3832085qA˘>
7˘4
*'
dense_input˙˙˙˙˙˙˙˙˙dŹ
p 

 
Ş "&#
unknown˙˙˙˙˙˙˙˙˙dŹĄ
,__inference_sequential_layer_call_fn_3832158qA˘>
7˘4
*'
dense_input˙˙˙˙˙˙˙˙˙dŹ
p

 
Ş "&#
unknown˙˙˙˙˙˙˙˙˙dŹ
,__inference_sequential_layer_call_fn_3834852l<˘9
2˘/
%"
inputs˙˙˙˙˙˙˙˙˙dŹ
p 

 
Ş "&#
unknown˙˙˙˙˙˙˙˙˙dŹ
,__inference_sequential_layer_call_fn_3834865l<˘9
2˘/
%"
inputs˙˙˙˙˙˙˙˙˙dŹ
p

 
Ş "&#
unknown˙˙˙˙˙˙˙˙˙dŹŞ
%__inference_signature_wrapper_3833772 %&'+,()*!"#$-.7˘4
˘ 
-Ş*
(
input_1
input_1˙˙˙˙˙˙˙˙˙"*Ş'
%
output_1
output_1Ż
Y__inference_token_and_position_embedding_layer_call_and_return_conditional_losses_3834083R&˘#
˘

x˙˙˙˙˙˙˙˙˙
Ş "$˘!

tensor_0	dŹ
 
>__inference_token_and_position_embedding_layer_call_fn_3834064G&˘#
˘

x˙˙˙˙˙˙˙˙˙
Ş "
unknown	dŹň
P__inference_transformer_block_1_layer_call_and_return_conditional_losses_3834593%&'+,()*!"#$c˘`
I˘F

inputsddŹ
"
context_sequence	dŹ
p
Ş

trainingp "(˘%

tensor_0ddŹ
 ň
P__inference_transformer_block_1_layer_call_and_return_conditional_losses_3834761%&'+,()*!"#$c˘`
I˘F

inputsddŹ
"
context_sequence	dŹ
p
Ş

trainingp"(˘%

tensor_0ddŹ
 Ě
5__inference_transformer_block_1_layer_call_fn_3834395%&'+,()*!"#$c˘`
I˘F

inputsddŹ
"
context_sequence	dŹ
p
Ş

trainingp "
unknownddŹĚ
5__inference_transformer_block_1_layer_call_fn_3834425%&'+,()*!"#$c˘`
I˘F

inputsddŹ
"
context_sequence	dŹ
p
Ş

trainingp"
unknownddŹČ
N__inference_transformer_block_layer_call_and_return_conditional_losses_3834247v	 ?˘<
%˘"

inputs	dŹ

 
p 
Ş

trainingp "(˘%

tensor_0ddŹ
 Č
N__inference_transformer_block_layer_call_and_return_conditional_losses_3834365v	 ?˘<
%˘"

inputs	dŹ

 
p 
Ş

trainingp"(˘%

tensor_0ddŹ
 ˘
3__inference_transformer_block_layer_call_fn_3834106k	 ?˘<
%˘"

inputs	dŹ

 
p 
Ş

trainingp "
unknownddŹ˘
3__inference_transformer_block_layer_call_fn_3834129k	 ?˘<
%˘"

inputs	dŹ

 
p 
Ş

trainingp"
unknownddŹ
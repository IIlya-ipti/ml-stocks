ÐÈ+
æ
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
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
®
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"

batch_dimsint "
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	

Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
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

Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
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
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
¥
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
.
Rsqrt
x"T
y"T"
Ttype:

2
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
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
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
G
SquaredDifference
x"T
y"T
z"T"
Ttype:

2	
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
2
StopGradient

input"T
output"T"	
Ttype
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
<
Sub
x"T
y"T
z"T"
Ttype:
2	
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
 "serve*2.10.02v2.10.0-rc3-6-g359c3cdfc5f8¬(
Ð
<Adam/private__transformer_block/layer_normalization_1/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *M
shared_name><Adam/private__transformer_block/layer_normalization_1/beta/v
É
PAdam/private__transformer_block/layer_normalization_1/beta/v/Read/ReadVariableOpReadVariableOp<Adam/private__transformer_block/layer_normalization_1/beta/v*
_output_shapes
: *
dtype0
Ò
=Adam/private__transformer_block/layer_normalization_1/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *N
shared_name?=Adam/private__transformer_block/layer_normalization_1/gamma/v
Ë
QAdam/private__transformer_block/layer_normalization_1/gamma/v/Read/ReadVariableOpReadVariableOp=Adam/private__transformer_block/layer_normalization_1/gamma/v*
_output_shapes
: *
dtype0
Ì
:Adam/private__transformer_block/layer_normalization/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *K
shared_name<:Adam/private__transformer_block/layer_normalization/beta/v
Å
NAdam/private__transformer_block/layer_normalization/beta/v/Read/ReadVariableOpReadVariableOp:Adam/private__transformer_block/layer_normalization/beta/v*
_output_shapes
: *
dtype0
Î
;Adam/private__transformer_block/layer_normalization/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *L
shared_name=;Adam/private__transformer_block/layer_normalization/gamma/v
Ç
OAdam/private__transformer_block/layer_normalization/gamma/v/Read/ReadVariableOpReadVariableOp;Adam/private__transformer_block/layer_normalization/gamma/v*
_output_shapes
: *
dtype0
~
Adam/dense_5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_nameAdam/dense_5/bias/v
w
'Adam/dense_5/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_5/bias/v*
_output_shapes
: *
dtype0

Adam/dense_5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *&
shared_nameAdam/dense_5/kernel/v

)Adam/dense_5/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_5/kernel/v*
_output_shapes

:  *
dtype0
~
Adam/dense_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_nameAdam/dense_4/bias/v
w
'Adam/dense_4/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_4/bias/v*
_output_shapes
: *
dtype0

Adam/dense_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *&
shared_nameAdam/dense_4/kernel/v

)Adam/dense_4/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_4/kernel/v*
_output_shapes

:  *
dtype0
ú
QAdam/private__transformer_block/private__multi_head_self_attention/dense_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *b
shared_nameSQAdam/private__transformer_block/private__multi_head_self_attention/dense_3/bias/v
ó
eAdam/private__transformer_block/private__multi_head_self_attention/dense_3/bias/v/Read/ReadVariableOpReadVariableOpQAdam/private__transformer_block/private__multi_head_self_attention/dense_3/bias/v*
_output_shapes
: *
dtype0

SAdam/private__transformer_block/private__multi_head_self_attention/dense_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *d
shared_nameUSAdam/private__transformer_block/private__multi_head_self_attention/dense_3/kernel/v
û
gAdam/private__transformer_block/private__multi_head_self_attention/dense_3/kernel/v/Read/ReadVariableOpReadVariableOpSAdam/private__transformer_block/private__multi_head_self_attention/dense_3/kernel/v*
_output_shapes

:  *
dtype0
ú
QAdam/private__transformer_block/private__multi_head_self_attention/dense_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *b
shared_nameSQAdam/private__transformer_block/private__multi_head_self_attention/dense_2/bias/v
ó
eAdam/private__transformer_block/private__multi_head_self_attention/dense_2/bias/v/Read/ReadVariableOpReadVariableOpQAdam/private__transformer_block/private__multi_head_self_attention/dense_2/bias/v*
_output_shapes
: *
dtype0

SAdam/private__transformer_block/private__multi_head_self_attention/dense_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *d
shared_nameUSAdam/private__transformer_block/private__multi_head_self_attention/dense_2/kernel/v
û
gAdam/private__transformer_block/private__multi_head_self_attention/dense_2/kernel/v/Read/ReadVariableOpReadVariableOpSAdam/private__transformer_block/private__multi_head_self_attention/dense_2/kernel/v*
_output_shapes

:  *
dtype0
ú
QAdam/private__transformer_block/private__multi_head_self_attention/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *b
shared_nameSQAdam/private__transformer_block/private__multi_head_self_attention/dense_1/bias/v
ó
eAdam/private__transformer_block/private__multi_head_self_attention/dense_1/bias/v/Read/ReadVariableOpReadVariableOpQAdam/private__transformer_block/private__multi_head_self_attention/dense_1/bias/v*
_output_shapes
: *
dtype0

SAdam/private__transformer_block/private__multi_head_self_attention/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *d
shared_nameUSAdam/private__transformer_block/private__multi_head_self_attention/dense_1/kernel/v
û
gAdam/private__transformer_block/private__multi_head_self_attention/dense_1/kernel/v/Read/ReadVariableOpReadVariableOpSAdam/private__transformer_block/private__multi_head_self_attention/dense_1/kernel/v*
_output_shapes

:  *
dtype0
ö
OAdam/private__transformer_block/private__multi_head_self_attention/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *`
shared_nameQOAdam/private__transformer_block/private__multi_head_self_attention/dense/bias/v
ï
cAdam/private__transformer_block/private__multi_head_self_attention/dense/bias/v/Read/ReadVariableOpReadVariableOpOAdam/private__transformer_block/private__multi_head_self_attention/dense/bias/v*
_output_shapes
: *
dtype0
þ
QAdam/private__transformer_block/private__multi_head_self_attention/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *b
shared_nameSQAdam/private__transformer_block/private__multi_head_self_attention/dense/kernel/v
÷
eAdam/private__transformer_block/private__multi_head_self_attention/dense/kernel/v/Read/ReadVariableOpReadVariableOpQAdam/private__transformer_block/private__multi_head_self_attention/dense/kernel/v*
_output_shapes

:  *
dtype0
ã
CAdam/private__token_and_position_embedding/embedding_1/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *T
shared_nameECAdam/private__token_and_position_embedding/embedding_1/embeddings/v
Ü
WAdam/private__token_and_position_embedding/embedding_1/embeddings/v/Read/ReadVariableOpReadVariableOpCAdam/private__token_and_position_embedding/embedding_1/embeddings/v*
_output_shapes
:	 *
dtype0
ß
AAdam/private__token_and_position_embedding/embedding/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Æ *R
shared_nameCAAdam/private__token_and_position_embedding/embedding/embeddings/v
Ø
UAdam/private__token_and_position_embedding/embedding/embeddings/v/Read/ReadVariableOpReadVariableOpAAdam/private__token_and_position_embedding/embedding/embeddings/v*
_output_shapes
:	Æ *
dtype0
~
Adam/dense_7/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_7/bias/v
w
'Adam/dense_7/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_7/bias/v*
_output_shapes
:*
dtype0

Adam/dense_7/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameAdam/dense_7/kernel/v

)Adam/dense_7/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_7/kernel/v*
_output_shapes

:*
dtype0
~
Adam/dense_6/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_6/bias/v
w
'Adam/dense_6/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_6/bias/v*
_output_shapes
:*
dtype0

Adam/dense_6/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *&
shared_nameAdam/dense_6/kernel/v

)Adam/dense_6/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_6/kernel/v*
_output_shapes

: *
dtype0
Ð
<Adam/private__transformer_block/layer_normalization_1/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *M
shared_name><Adam/private__transformer_block/layer_normalization_1/beta/m
É
PAdam/private__transformer_block/layer_normalization_1/beta/m/Read/ReadVariableOpReadVariableOp<Adam/private__transformer_block/layer_normalization_1/beta/m*
_output_shapes
: *
dtype0
Ò
=Adam/private__transformer_block/layer_normalization_1/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *N
shared_name?=Adam/private__transformer_block/layer_normalization_1/gamma/m
Ë
QAdam/private__transformer_block/layer_normalization_1/gamma/m/Read/ReadVariableOpReadVariableOp=Adam/private__transformer_block/layer_normalization_1/gamma/m*
_output_shapes
: *
dtype0
Ì
:Adam/private__transformer_block/layer_normalization/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *K
shared_name<:Adam/private__transformer_block/layer_normalization/beta/m
Å
NAdam/private__transformer_block/layer_normalization/beta/m/Read/ReadVariableOpReadVariableOp:Adam/private__transformer_block/layer_normalization/beta/m*
_output_shapes
: *
dtype0
Î
;Adam/private__transformer_block/layer_normalization/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *L
shared_name=;Adam/private__transformer_block/layer_normalization/gamma/m
Ç
OAdam/private__transformer_block/layer_normalization/gamma/m/Read/ReadVariableOpReadVariableOp;Adam/private__transformer_block/layer_normalization/gamma/m*
_output_shapes
: *
dtype0
~
Adam/dense_5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_nameAdam/dense_5/bias/m
w
'Adam/dense_5/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_5/bias/m*
_output_shapes
: *
dtype0

Adam/dense_5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *&
shared_nameAdam/dense_5/kernel/m

)Adam/dense_5/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_5/kernel/m*
_output_shapes

:  *
dtype0
~
Adam/dense_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_nameAdam/dense_4/bias/m
w
'Adam/dense_4/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_4/bias/m*
_output_shapes
: *
dtype0

Adam/dense_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *&
shared_nameAdam/dense_4/kernel/m

)Adam/dense_4/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_4/kernel/m*
_output_shapes

:  *
dtype0
ú
QAdam/private__transformer_block/private__multi_head_self_attention/dense_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *b
shared_nameSQAdam/private__transformer_block/private__multi_head_self_attention/dense_3/bias/m
ó
eAdam/private__transformer_block/private__multi_head_self_attention/dense_3/bias/m/Read/ReadVariableOpReadVariableOpQAdam/private__transformer_block/private__multi_head_self_attention/dense_3/bias/m*
_output_shapes
: *
dtype0

SAdam/private__transformer_block/private__multi_head_self_attention/dense_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *d
shared_nameUSAdam/private__transformer_block/private__multi_head_self_attention/dense_3/kernel/m
û
gAdam/private__transformer_block/private__multi_head_self_attention/dense_3/kernel/m/Read/ReadVariableOpReadVariableOpSAdam/private__transformer_block/private__multi_head_self_attention/dense_3/kernel/m*
_output_shapes

:  *
dtype0
ú
QAdam/private__transformer_block/private__multi_head_self_attention/dense_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *b
shared_nameSQAdam/private__transformer_block/private__multi_head_self_attention/dense_2/bias/m
ó
eAdam/private__transformer_block/private__multi_head_self_attention/dense_2/bias/m/Read/ReadVariableOpReadVariableOpQAdam/private__transformer_block/private__multi_head_self_attention/dense_2/bias/m*
_output_shapes
: *
dtype0

SAdam/private__transformer_block/private__multi_head_self_attention/dense_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *d
shared_nameUSAdam/private__transformer_block/private__multi_head_self_attention/dense_2/kernel/m
û
gAdam/private__transformer_block/private__multi_head_self_attention/dense_2/kernel/m/Read/ReadVariableOpReadVariableOpSAdam/private__transformer_block/private__multi_head_self_attention/dense_2/kernel/m*
_output_shapes

:  *
dtype0
ú
QAdam/private__transformer_block/private__multi_head_self_attention/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *b
shared_nameSQAdam/private__transformer_block/private__multi_head_self_attention/dense_1/bias/m
ó
eAdam/private__transformer_block/private__multi_head_self_attention/dense_1/bias/m/Read/ReadVariableOpReadVariableOpQAdam/private__transformer_block/private__multi_head_self_attention/dense_1/bias/m*
_output_shapes
: *
dtype0

SAdam/private__transformer_block/private__multi_head_self_attention/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *d
shared_nameUSAdam/private__transformer_block/private__multi_head_self_attention/dense_1/kernel/m
û
gAdam/private__transformer_block/private__multi_head_self_attention/dense_1/kernel/m/Read/ReadVariableOpReadVariableOpSAdam/private__transformer_block/private__multi_head_self_attention/dense_1/kernel/m*
_output_shapes

:  *
dtype0
ö
OAdam/private__transformer_block/private__multi_head_self_attention/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *`
shared_nameQOAdam/private__transformer_block/private__multi_head_self_attention/dense/bias/m
ï
cAdam/private__transformer_block/private__multi_head_self_attention/dense/bias/m/Read/ReadVariableOpReadVariableOpOAdam/private__transformer_block/private__multi_head_self_attention/dense/bias/m*
_output_shapes
: *
dtype0
þ
QAdam/private__transformer_block/private__multi_head_self_attention/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *b
shared_nameSQAdam/private__transformer_block/private__multi_head_self_attention/dense/kernel/m
÷
eAdam/private__transformer_block/private__multi_head_self_attention/dense/kernel/m/Read/ReadVariableOpReadVariableOpQAdam/private__transformer_block/private__multi_head_self_attention/dense/kernel/m*
_output_shapes

:  *
dtype0
ã
CAdam/private__token_and_position_embedding/embedding_1/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *T
shared_nameECAdam/private__token_and_position_embedding/embedding_1/embeddings/m
Ü
WAdam/private__token_and_position_embedding/embedding_1/embeddings/m/Read/ReadVariableOpReadVariableOpCAdam/private__token_and_position_embedding/embedding_1/embeddings/m*
_output_shapes
:	 *
dtype0
ß
AAdam/private__token_and_position_embedding/embedding/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Æ *R
shared_nameCAAdam/private__token_and_position_embedding/embedding/embeddings/m
Ø
UAdam/private__token_and_position_embedding/embedding/embeddings/m/Read/ReadVariableOpReadVariableOpAAdam/private__token_and_position_embedding/embedding/embeddings/m*
_output_shapes
:	Æ *
dtype0
~
Adam/dense_7/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_7/bias/m
w
'Adam/dense_7/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_7/bias/m*
_output_shapes
:*
dtype0

Adam/dense_7/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameAdam/dense_7/kernel/m

)Adam/dense_7/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_7/kernel/m*
_output_shapes

:*
dtype0
~
Adam/dense_6/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_6/bias/m
w
'Adam/dense_6/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_6/bias/m*
_output_shapes
:*
dtype0

Adam/dense_6/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *&
shared_nameAdam/dense_6/kernel/m

)Adam/dense_6/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_6/kernel/m*
_output_shapes

: *
dtype0
w
false_negativesVarHandleOp*
_output_shapes
: *
dtype0*
shape:È* 
shared_namefalse_negatives
p
#false_negatives/Read/ReadVariableOpReadVariableOpfalse_negatives*
_output_shapes	
:È*
dtype0
w
false_positivesVarHandleOp*
_output_shapes
: *
dtype0*
shape:È* 
shared_namefalse_positives
p
#false_positives/Read/ReadVariableOpReadVariableOpfalse_positives*
_output_shapes	
:È*
dtype0
u
true_negativesVarHandleOp*
_output_shapes
: *
dtype0*
shape:È*
shared_nametrue_negatives
n
"true_negatives/Read/ReadVariableOpReadVariableOptrue_negatives*
_output_shapes	
:È*
dtype0
u
true_positivesVarHandleOp*
_output_shapes
: *
dtype0*
shape:È*
shared_nametrue_positives
n
"true_positives/Read/ReadVariableOpReadVariableOptrue_positives*
_output_shapes	
:È*
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
Â
5private__transformer_block/layer_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *F
shared_name75private__transformer_block/layer_normalization_1/beta
»
Iprivate__transformer_block/layer_normalization_1/beta/Read/ReadVariableOpReadVariableOp5private__transformer_block/layer_normalization_1/beta*
_output_shapes
: *
dtype0
Ä
6private__transformer_block/layer_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *G
shared_name86private__transformer_block/layer_normalization_1/gamma
½
Jprivate__transformer_block/layer_normalization_1/gamma/Read/ReadVariableOpReadVariableOp6private__transformer_block/layer_normalization_1/gamma*
_output_shapes
: *
dtype0
¾
3private__transformer_block/layer_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *D
shared_name53private__transformer_block/layer_normalization/beta
·
Gprivate__transformer_block/layer_normalization/beta/Read/ReadVariableOpReadVariableOp3private__transformer_block/layer_normalization/beta*
_output_shapes
: *
dtype0
À
4private__transformer_block/layer_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *E
shared_name64private__transformer_block/layer_normalization/gamma
¹
Hprivate__transformer_block/layer_normalization/gamma/Read/ReadVariableOpReadVariableOp4private__transformer_block/layer_normalization/gamma*
_output_shapes
: *
dtype0
p
dense_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_5/bias
i
 dense_5/bias/Read/ReadVariableOpReadVariableOpdense_5/bias*
_output_shapes
: *
dtype0
x
dense_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *
shared_namedense_5/kernel
q
"dense_5/kernel/Read/ReadVariableOpReadVariableOpdense_5/kernel*
_output_shapes

:  *
dtype0
p
dense_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_4/bias
i
 dense_4/bias/Read/ReadVariableOpReadVariableOpdense_4/bias*
_output_shapes
: *
dtype0
x
dense_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *
shared_namedense_4/kernel
q
"dense_4/kernel/Read/ReadVariableOpReadVariableOpdense_4/kernel*
_output_shapes

:  *
dtype0
ì
Jprivate__transformer_block/private__multi_head_self_attention/dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *[
shared_nameLJprivate__transformer_block/private__multi_head_self_attention/dense_3/bias
å
^private__transformer_block/private__multi_head_self_attention/dense_3/bias/Read/ReadVariableOpReadVariableOpJprivate__transformer_block/private__multi_head_self_attention/dense_3/bias*
_output_shapes
: *
dtype0
ô
Lprivate__transformer_block/private__multi_head_self_attention/dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *]
shared_nameNLprivate__transformer_block/private__multi_head_self_attention/dense_3/kernel
í
`private__transformer_block/private__multi_head_self_attention/dense_3/kernel/Read/ReadVariableOpReadVariableOpLprivate__transformer_block/private__multi_head_self_attention/dense_3/kernel*
_output_shapes

:  *
dtype0
ì
Jprivate__transformer_block/private__multi_head_self_attention/dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *[
shared_nameLJprivate__transformer_block/private__multi_head_self_attention/dense_2/bias
å
^private__transformer_block/private__multi_head_self_attention/dense_2/bias/Read/ReadVariableOpReadVariableOpJprivate__transformer_block/private__multi_head_self_attention/dense_2/bias*
_output_shapes
: *
dtype0
ô
Lprivate__transformer_block/private__multi_head_self_attention/dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *]
shared_nameNLprivate__transformer_block/private__multi_head_self_attention/dense_2/kernel
í
`private__transformer_block/private__multi_head_self_attention/dense_2/kernel/Read/ReadVariableOpReadVariableOpLprivate__transformer_block/private__multi_head_self_attention/dense_2/kernel*
_output_shapes

:  *
dtype0
ì
Jprivate__transformer_block/private__multi_head_self_attention/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *[
shared_nameLJprivate__transformer_block/private__multi_head_self_attention/dense_1/bias
å
^private__transformer_block/private__multi_head_self_attention/dense_1/bias/Read/ReadVariableOpReadVariableOpJprivate__transformer_block/private__multi_head_self_attention/dense_1/bias*
_output_shapes
: *
dtype0
ô
Lprivate__transformer_block/private__multi_head_self_attention/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *]
shared_nameNLprivate__transformer_block/private__multi_head_self_attention/dense_1/kernel
í
`private__transformer_block/private__multi_head_self_attention/dense_1/kernel/Read/ReadVariableOpReadVariableOpLprivate__transformer_block/private__multi_head_self_attention/dense_1/kernel*
_output_shapes

:  *
dtype0
è
Hprivate__transformer_block/private__multi_head_self_attention/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *Y
shared_nameJHprivate__transformer_block/private__multi_head_self_attention/dense/bias
á
\private__transformer_block/private__multi_head_self_attention/dense/bias/Read/ReadVariableOpReadVariableOpHprivate__transformer_block/private__multi_head_self_attention/dense/bias*
_output_shapes
: *
dtype0
ð
Jprivate__transformer_block/private__multi_head_self_attention/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *[
shared_nameLJprivate__transformer_block/private__multi_head_self_attention/dense/kernel
é
^private__transformer_block/private__multi_head_self_attention/dense/kernel/Read/ReadVariableOpReadVariableOpJprivate__transformer_block/private__multi_head_self_attention/dense/kernel*
_output_shapes

:  *
dtype0
Õ
<private__token_and_position_embedding/embedding_1/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *M
shared_name><private__token_and_position_embedding/embedding_1/embeddings
Î
Pprivate__token_and_position_embedding/embedding_1/embeddings/Read/ReadVariableOpReadVariableOp<private__token_and_position_embedding/embedding_1/embeddings*
_output_shapes
:	 *
dtype0
Ñ
:private__token_and_position_embedding/embedding/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Æ *K
shared_name<:private__token_and_position_embedding/embedding/embeddings
Ê
Nprivate__token_and_position_embedding/embedding/embeddings/Read/ReadVariableOpReadVariableOp:private__token_and_position_embedding/embedding/embeddings*
_output_shapes
:	Æ *
dtype0
p
dense_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_7/bias
i
 dense_7/bias/Read/ReadVariableOpReadVariableOpdense_7/bias*
_output_shapes
:*
dtype0
x
dense_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namedense_7/kernel
q
"dense_7/kernel/Read/ReadVariableOpReadVariableOpdense_7/kernel*
_output_shapes

:*
dtype0
p
dense_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_6/bias
i
 dense_6/bias/Read/ReadVariableOpReadVariableOpdense_6/bias*
_output_shapes
:*
dtype0
x
dense_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *
shared_namedense_6/kernel
q
"dense_6/kernel/Read/ReadVariableOpReadVariableOpdense_6/kernel*
_output_shapes

: *
dtype0
|
serving_default_input_1Placeholder*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ

StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1<private__token_and_position_embedding/embedding_1/embeddings:private__token_and_position_embedding/embedding/embeddingsJprivate__transformer_block/private__multi_head_self_attention/dense/kernelHprivate__transformer_block/private__multi_head_self_attention/dense/biasLprivate__transformer_block/private__multi_head_self_attention/dense_1/kernelJprivate__transformer_block/private__multi_head_self_attention/dense_1/biasLprivate__transformer_block/private__multi_head_self_attention/dense_2/kernelJprivate__transformer_block/private__multi_head_self_attention/dense_2/biasLprivate__transformer_block/private__multi_head_self_attention/dense_3/kernelJprivate__transformer_block/private__multi_head_self_attention/dense_3/bias4private__transformer_block/layer_normalization/gamma3private__transformer_block/layer_normalization/betadense_4/kerneldense_4/biasdense_5/kerneldense_5/bias6private__transformer_block/layer_normalization_1/gamma5private__transformer_block/layer_normalization_1/betadense_6/kerneldense_6/biasdense_7/kerneldense_7/bias*"
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*8
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *+
f&R$
"__inference_signature_wrapper_6078

NoOpNoOp
ÁÀ
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*û¿
valueð¿Bì¿ Bä¿

layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer-4
layer_with_weights-2
layer-5
layer-6
layer_with_weights-3
layer-7
		variables

trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*
* 
¬
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
	token_emb
pos_emb*
Þ
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
 att
!ffn
"
layernorm1
#
layernorm2
$dropout1
%dropout2*

&	variables
'trainable_variables
(regularization_losses
)	keras_api
*__call__
*+&call_and_return_all_conditional_losses* 
¥
,	variables
-trainable_variables
.regularization_losses
/	keras_api
0__call__
*1&call_and_return_all_conditional_losses
2_random_generator* 
¦
3	variables
4trainable_variables
5regularization_losses
6	keras_api
7__call__
*8&call_and_return_all_conditional_losses

9kernel
:bias*
¥
;	variables
<trainable_variables
=regularization_losses
>	keras_api
?__call__
*@&call_and_return_all_conditional_losses
A_random_generator* 
¦
B	variables
Ctrainable_variables
Dregularization_losses
E	keras_api
F__call__
*G&call_and_return_all_conditional_losses

Hkernel
Ibias*
ª
J0
K1
L2
M3
N4
O5
P6
Q7
R8
S9
T10
U11
V12
W13
X14
Y15
Z16
[17
918
:19
H20
I21*
ª
J0
K1
L2
M3
N4
O5
P6
Q7
R8
S9
T10
U11
V12
W13
X14
Y15
Z16
[17
918
:19
H20
I21*
* 
°
\non_trainable_variables

]layers
^metrics
_layer_regularization_losses
`layer_metrics
		variables

trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
atrace_0
btrace_1
ctrace_2
dtrace_3* 
6
etrace_0
ftrace_1
gtrace_2
htrace_3* 
* 
ü
iiter

jbeta_1

kbeta_2
	ldecay
mlearning_rate9mâ:mãHmäImåJmæKmçLmèMméNmêOmëPmìQmíRmîSmïTmðUmñVmòWmóXmôYmõZmö[m÷9vø:vùHvúIvûJvüKvýLvþMvÿNvOvPvQvRvSvTvUvVvWvXvYvZv[v*

nserving_default* 

J0
K1*

J0
K1*
* 

onon_trainable_variables

players
qmetrics
rlayer_regularization_losses
slayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

ttrace_0* 

utrace_0* 
 
v	variables
wtrainable_variables
xregularization_losses
y	keras_api
z__call__
*{&call_and_return_all_conditional_losses
J
embeddings*
¢
|	variables
}trainable_variables
~regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
K
embeddings*
z
L0
M1
N2
O3
P4
Q5
R6
S7
T8
U9
V10
W11
X12
Y13
Z14
[15*
z
L0
M1
N2
O3
P4
Q5
R6
S7
T8
U9
V10
W11
X12
Y13
Z14
[15*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

trace_0
trace_1* 

trace_0
trace_1* 
Þ
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
query_dense
	key_dense
value_dense
combine_heads*
è
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses*
¶
	variables
trainable_variables
regularization_losses
 	keras_api
¡__call__
+¢&call_and_return_all_conditional_losses
	£axis
	Xgamma
Ybeta*
¶
¤	variables
¥trainable_variables
¦regularization_losses
§	keras_api
¨__call__
+©&call_and_return_all_conditional_losses
	ªaxis
	Zgamma
[beta*
¬
«	variables
¬trainable_variables
­regularization_losses
®	keras_api
¯__call__
+°&call_and_return_all_conditional_losses
±_random_generator* 
¬
²	variables
³trainable_variables
´regularization_losses
µ	keras_api
¶__call__
+·&call_and_return_all_conditional_losses
¸_random_generator* 
* 
* 
* 

¹non_trainable_variables
ºlayers
»metrics
 ¼layer_regularization_losses
½layer_metrics
&	variables
'trainable_variables
(regularization_losses
*__call__
*+&call_and_return_all_conditional_losses
&+"call_and_return_conditional_losses* 

¾trace_0* 

¿trace_0* 
* 
* 
* 

Ànon_trainable_variables
Álayers
Âmetrics
 Ãlayer_regularization_losses
Älayer_metrics
,	variables
-trainable_variables
.regularization_losses
0__call__
*1&call_and_return_all_conditional_losses
&1"call_and_return_conditional_losses* 

Åtrace_0
Ætrace_1* 

Çtrace_0
Ètrace_1* 
* 

90
:1*

90
:1*
* 

Énon_trainable_variables
Êlayers
Ëmetrics
 Ìlayer_regularization_losses
Ílayer_metrics
3	variables
4trainable_variables
5regularization_losses
7__call__
*8&call_and_return_all_conditional_losses
&8"call_and_return_conditional_losses*

Îtrace_0* 

Ïtrace_0* 
^X
VARIABLE_VALUEdense_6/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_6/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 

Ðnon_trainable_variables
Ñlayers
Òmetrics
 Ólayer_regularization_losses
Ôlayer_metrics
;	variables
<trainable_variables
=regularization_losses
?__call__
*@&call_and_return_all_conditional_losses
&@"call_and_return_conditional_losses* 

Õtrace_0
Ötrace_1* 

×trace_0
Øtrace_1* 
* 

H0
I1*

H0
I1*
* 

Ùnon_trainable_variables
Úlayers
Ûmetrics
 Ülayer_regularization_losses
Ýlayer_metrics
B	variables
Ctrainable_variables
Dregularization_losses
F__call__
*G&call_and_return_all_conditional_losses
&G"call_and_return_conditional_losses*

Þtrace_0* 

ßtrace_0* 
^X
VARIABLE_VALUEdense_7/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_7/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*
zt
VARIABLE_VALUE:private__token_and_position_embedding/embedding/embeddings&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE<private__token_and_position_embedding/embedding_1/embeddings&variables/1/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEJprivate__transformer_block/private__multi_head_self_attention/dense/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEHprivate__transformer_block/private__multi_head_self_attention/dense/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUELprivate__transformer_block/private__multi_head_self_attention/dense_1/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEJprivate__transformer_block/private__multi_head_self_attention/dense_1/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUELprivate__transformer_block/private__multi_head_self_attention/dense_2/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEJprivate__transformer_block/private__multi_head_self_attention/dense_2/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUELprivate__transformer_block/private__multi_head_self_attention/dense_3/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEJprivate__transformer_block/private__multi_head_self_attention/dense_3/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEdense_4/kernel'variables/10/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEdense_4/bias'variables/11/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEdense_5/kernel'variables/12/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEdense_5/bias'variables/13/.ATTRIBUTES/VARIABLE_VALUE*
uo
VARIABLE_VALUE4private__transformer_block/layer_normalization/gamma'variables/14/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUE3private__transformer_block/layer_normalization/beta'variables/15/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE6private__transformer_block/layer_normalization_1/gamma'variables/16/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE5private__transformer_block/layer_normalization_1/beta'variables/17/.ATTRIBUTES/VARIABLE_VALUE*
* 
<
0
1
2
3
4
5
6
7*

à0
á1*
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
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 

0
1*
* 
* 
* 
* 
* 

J0*

J0*
* 

ânon_trainable_variables
ãlayers
ämetrics
 ålayer_regularization_losses
ælayer_metrics
v	variables
wtrainable_variables
xregularization_losses
z__call__
*{&call_and_return_all_conditional_losses
&{"call_and_return_conditional_losses*
* 
* 

K0*

K0*
* 

çnon_trainable_variables
èlayers
émetrics
 êlayer_regularization_losses
ëlayer_metrics
|	variables
}trainable_variables
~regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*
* 
* 
* 
.
 0
!1
"2
#3
$4
%5*
* 
* 
* 
* 
* 
* 
* 
<
L0
M1
N2
O3
P4
Q5
R6
S7*
<
L0
M1
N2
O3
P4
Q5
R6
S7*
* 

ìnon_trainable_variables
ílayers
îmetrics
 ïlayer_regularization_losses
ðlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*
* 
* 
¬
ñ	variables
òtrainable_variables
óregularization_losses
ô	keras_api
õ__call__
+ö&call_and_return_all_conditional_losses

Lkernel
Mbias*
¬
÷	variables
øtrainable_variables
ùregularization_losses
ú	keras_api
û__call__
+ü&call_and_return_all_conditional_losses

Nkernel
Obias*
¬
ý	variables
þtrainable_variables
ÿregularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses

Pkernel
Qbias*
¬
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses

Rkernel
Sbias*
¬
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses

Tkernel
Ubias*
¬
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses

Vkernel
Wbias*
 
T0
U1
V2
W3*
 
T0
U1
V2
W3*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*
:
trace_0
trace_1
trace_2
trace_3* 
:
trace_0
trace_1
 trace_2
¡trace_3* 

X0
Y1*

X0
Y1*
* 

¢non_trainable_variables
£layers
¤metrics
 ¥layer_regularization_losses
¦layer_metrics
	variables
trainable_variables
regularization_losses
¡__call__
+¢&call_and_return_all_conditional_losses
'¢"call_and_return_conditional_losses*
* 
* 
* 

Z0
[1*

Z0
[1*
* 

§non_trainable_variables
¨layers
©metrics
 ªlayer_regularization_losses
«layer_metrics
¤	variables
¥trainable_variables
¦regularization_losses
¨__call__
+©&call_and_return_all_conditional_losses
'©"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 

¬non_trainable_variables
­layers
®metrics
 ¯layer_regularization_losses
°layer_metrics
«	variables
¬trainable_variables
­regularization_losses
¯__call__
+°&call_and_return_all_conditional_losses
'°"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
* 

±non_trainable_variables
²layers
³metrics
 ´layer_regularization_losses
µlayer_metrics
²	variables
³trainable_variables
´regularization_losses
¶__call__
+·&call_and_return_all_conditional_losses
'·"call_and_return_conditional_losses* 
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
<
¶	variables
·	keras_api

¸total

¹count*
z
º	variables
»	keras_api
¼true_positives
½true_negatives
¾false_positives
¿false_negatives*
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
$
0
1
2
3*
* 
* 
* 

L0
M1*

L0
M1*
* 

Ànon_trainable_variables
Álayers
Âmetrics
 Ãlayer_regularization_losses
Älayer_metrics
ñ	variables
òtrainable_variables
óregularization_losses
õ__call__
+ö&call_and_return_all_conditional_losses
'ö"call_and_return_conditional_losses*
* 
* 

N0
O1*

N0
O1*
* 

Ånon_trainable_variables
Ælayers
Çmetrics
 Èlayer_regularization_losses
Élayer_metrics
÷	variables
øtrainable_variables
ùregularization_losses
û__call__
+ü&call_and_return_all_conditional_losses
'ü"call_and_return_conditional_losses*
* 
* 

P0
Q1*

P0
Q1*
* 

Ênon_trainable_variables
Ëlayers
Ìmetrics
 Ílayer_regularization_losses
Îlayer_metrics
ý	variables
þtrainable_variables
ÿregularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*
* 
* 

R0
S1*

R0
S1*
* 

Ïnon_trainable_variables
Ðlayers
Ñmetrics
 Òlayer_regularization_losses
Ólayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*
* 
* 

T0
U1*

T0
U1*
* 

Ônon_trainable_variables
Õlayers
Ömetrics
 ×layer_regularization_losses
Ølayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*

Ùtrace_0* 

Útrace_0* 

V0
W1*

V0
W1*
* 

Ûnon_trainable_variables
Ülayers
Ýmetrics
 Þlayer_regularization_losses
ßlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*

àtrace_0* 

átrace_0* 
* 

0
1*
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

¸0
¹1*

¶	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*
$
¼0
½1
¾2
¿3*

º	variables*
e_
VARIABLE_VALUEtrue_positives=keras_api/metrics/1/true_positives/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUEtrue_negatives=keras_api/metrics/1/true_negatives/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUEfalse_positives>keras_api/metrics/1/false_positives/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUEfalse_negatives>keras_api/metrics/1/false_negatives/.ATTRIBUTES/VARIABLE_VALUE*
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
{
VARIABLE_VALUEAdam/dense_6/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_6/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUEAdam/dense_7/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_7/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAAdam/private__token_and_position_embedding/embedding/embeddings/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
 
VARIABLE_VALUECAdam/private__token_and_position_embedding/embedding_1/embeddings/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
®§
VARIABLE_VALUEQAdam/private__transformer_block/private__multi_head_self_attention/dense/kernel/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
¬¥
VARIABLE_VALUEOAdam/private__transformer_block/private__multi_head_self_attention/dense/bias/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
°©
VARIABLE_VALUESAdam/private__transformer_block/private__multi_head_self_attention/dense_1/kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
®§
VARIABLE_VALUEQAdam/private__transformer_block/private__multi_head_self_attention/dense_1/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
°©
VARIABLE_VALUESAdam/private__transformer_block/private__multi_head_self_attention/dense_2/kernel/mBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
®§
VARIABLE_VALUEQAdam/private__transformer_block/private__multi_head_self_attention/dense_2/bias/mBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
°©
VARIABLE_VALUESAdam/private__transformer_block/private__multi_head_self_attention/dense_3/kernel/mBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
®§
VARIABLE_VALUEQAdam/private__transformer_block/private__multi_head_self_attention/dense_3/bias/mBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEAdam/dense_4/kernel/mCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUEAdam/dense_4/bias/mCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEAdam/dense_5/kernel/mCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUEAdam/dense_5/bias/mCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE;Adam/private__transformer_block/layer_normalization/gamma/mCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE:Adam/private__transformer_block/layer_normalization/beta/mCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE=Adam/private__transformer_block/layer_normalization_1/gamma/mCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE<Adam/private__transformer_block/layer_normalization_1/beta/mCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUEAdam/dense_6/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_6/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUEAdam/dense_7/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_7/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAAdam/private__token_and_position_embedding/embedding/embeddings/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
 
VARIABLE_VALUECAdam/private__token_and_position_embedding/embedding_1/embeddings/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
®§
VARIABLE_VALUEQAdam/private__transformer_block/private__multi_head_self_attention/dense/kernel/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
¬¥
VARIABLE_VALUEOAdam/private__transformer_block/private__multi_head_self_attention/dense/bias/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
°©
VARIABLE_VALUESAdam/private__transformer_block/private__multi_head_self_attention/dense_1/kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
®§
VARIABLE_VALUEQAdam/private__transformer_block/private__multi_head_self_attention/dense_1/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
°©
VARIABLE_VALUESAdam/private__transformer_block/private__multi_head_self_attention/dense_2/kernel/vBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
®§
VARIABLE_VALUEQAdam/private__transformer_block/private__multi_head_self_attention/dense_2/bias/vBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
°©
VARIABLE_VALUESAdam/private__transformer_block/private__multi_head_self_attention/dense_3/kernel/vBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
®§
VARIABLE_VALUEQAdam/private__transformer_block/private__multi_head_self_attention/dense_3/bias/vBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEAdam/dense_4/kernel/vCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUEAdam/dense_4/bias/vCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEAdam/dense_5/kernel/vCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUEAdam/dense_5/bias/vCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE;Adam/private__transformer_block/layer_normalization/gamma/vCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE:Adam/private__transformer_block/layer_normalization/beta/vCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE=Adam/private__transformer_block/layer_normalization_1/gamma/vCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE<Adam/private__transformer_block/layer_normalization_1/beta/vCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Ã+
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename"dense_6/kernel/Read/ReadVariableOp dense_6/bias/Read/ReadVariableOp"dense_7/kernel/Read/ReadVariableOp dense_7/bias/Read/ReadVariableOpNprivate__token_and_position_embedding/embedding/embeddings/Read/ReadVariableOpPprivate__token_and_position_embedding/embedding_1/embeddings/Read/ReadVariableOp^private__transformer_block/private__multi_head_self_attention/dense/kernel/Read/ReadVariableOp\private__transformer_block/private__multi_head_self_attention/dense/bias/Read/ReadVariableOp`private__transformer_block/private__multi_head_self_attention/dense_1/kernel/Read/ReadVariableOp^private__transformer_block/private__multi_head_self_attention/dense_1/bias/Read/ReadVariableOp`private__transformer_block/private__multi_head_self_attention/dense_2/kernel/Read/ReadVariableOp^private__transformer_block/private__multi_head_self_attention/dense_2/bias/Read/ReadVariableOp`private__transformer_block/private__multi_head_self_attention/dense_3/kernel/Read/ReadVariableOp^private__transformer_block/private__multi_head_self_attention/dense_3/bias/Read/ReadVariableOp"dense_4/kernel/Read/ReadVariableOp dense_4/bias/Read/ReadVariableOp"dense_5/kernel/Read/ReadVariableOp dense_5/bias/Read/ReadVariableOpHprivate__transformer_block/layer_normalization/gamma/Read/ReadVariableOpGprivate__transformer_block/layer_normalization/beta/Read/ReadVariableOpJprivate__transformer_block/layer_normalization_1/gamma/Read/ReadVariableOpIprivate__transformer_block/layer_normalization_1/beta/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp"true_positives/Read/ReadVariableOp"true_negatives/Read/ReadVariableOp#false_positives/Read/ReadVariableOp#false_negatives/Read/ReadVariableOp)Adam/dense_6/kernel/m/Read/ReadVariableOp'Adam/dense_6/bias/m/Read/ReadVariableOp)Adam/dense_7/kernel/m/Read/ReadVariableOp'Adam/dense_7/bias/m/Read/ReadVariableOpUAdam/private__token_and_position_embedding/embedding/embeddings/m/Read/ReadVariableOpWAdam/private__token_and_position_embedding/embedding_1/embeddings/m/Read/ReadVariableOpeAdam/private__transformer_block/private__multi_head_self_attention/dense/kernel/m/Read/ReadVariableOpcAdam/private__transformer_block/private__multi_head_self_attention/dense/bias/m/Read/ReadVariableOpgAdam/private__transformer_block/private__multi_head_self_attention/dense_1/kernel/m/Read/ReadVariableOpeAdam/private__transformer_block/private__multi_head_self_attention/dense_1/bias/m/Read/ReadVariableOpgAdam/private__transformer_block/private__multi_head_self_attention/dense_2/kernel/m/Read/ReadVariableOpeAdam/private__transformer_block/private__multi_head_self_attention/dense_2/bias/m/Read/ReadVariableOpgAdam/private__transformer_block/private__multi_head_self_attention/dense_3/kernel/m/Read/ReadVariableOpeAdam/private__transformer_block/private__multi_head_self_attention/dense_3/bias/m/Read/ReadVariableOp)Adam/dense_4/kernel/m/Read/ReadVariableOp'Adam/dense_4/bias/m/Read/ReadVariableOp)Adam/dense_5/kernel/m/Read/ReadVariableOp'Adam/dense_5/bias/m/Read/ReadVariableOpOAdam/private__transformer_block/layer_normalization/gamma/m/Read/ReadVariableOpNAdam/private__transformer_block/layer_normalization/beta/m/Read/ReadVariableOpQAdam/private__transformer_block/layer_normalization_1/gamma/m/Read/ReadVariableOpPAdam/private__transformer_block/layer_normalization_1/beta/m/Read/ReadVariableOp)Adam/dense_6/kernel/v/Read/ReadVariableOp'Adam/dense_6/bias/v/Read/ReadVariableOp)Adam/dense_7/kernel/v/Read/ReadVariableOp'Adam/dense_7/bias/v/Read/ReadVariableOpUAdam/private__token_and_position_embedding/embedding/embeddings/v/Read/ReadVariableOpWAdam/private__token_and_position_embedding/embedding_1/embeddings/v/Read/ReadVariableOpeAdam/private__transformer_block/private__multi_head_self_attention/dense/kernel/v/Read/ReadVariableOpcAdam/private__transformer_block/private__multi_head_self_attention/dense/bias/v/Read/ReadVariableOpgAdam/private__transformer_block/private__multi_head_self_attention/dense_1/kernel/v/Read/ReadVariableOpeAdam/private__transformer_block/private__multi_head_self_attention/dense_1/bias/v/Read/ReadVariableOpgAdam/private__transformer_block/private__multi_head_self_attention/dense_2/kernel/v/Read/ReadVariableOpeAdam/private__transformer_block/private__multi_head_self_attention/dense_2/bias/v/Read/ReadVariableOpgAdam/private__transformer_block/private__multi_head_self_attention/dense_3/kernel/v/Read/ReadVariableOpeAdam/private__transformer_block/private__multi_head_self_attention/dense_3/bias/v/Read/ReadVariableOp)Adam/dense_4/kernel/v/Read/ReadVariableOp'Adam/dense_4/bias/v/Read/ReadVariableOp)Adam/dense_5/kernel/v/Read/ReadVariableOp'Adam/dense_5/bias/v/Read/ReadVariableOpOAdam/private__transformer_block/layer_normalization/gamma/v/Read/ReadVariableOpNAdam/private__transformer_block/layer_normalization/beta/v/Read/ReadVariableOpQAdam/private__transformer_block/layer_normalization_1/gamma/v/Read/ReadVariableOpPAdam/private__transformer_block/layer_normalization_1/beta/v/Read/ReadVariableOpConst*Z
TinS
Q2O	*
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
GPU 2J 8 *&
f!R
__inference__traced_save_7955
º
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_6/kerneldense_6/biasdense_7/kerneldense_7/bias:private__token_and_position_embedding/embedding/embeddings<private__token_and_position_embedding/embedding_1/embeddingsJprivate__transformer_block/private__multi_head_self_attention/dense/kernelHprivate__transformer_block/private__multi_head_self_attention/dense/biasLprivate__transformer_block/private__multi_head_self_attention/dense_1/kernelJprivate__transformer_block/private__multi_head_self_attention/dense_1/biasLprivate__transformer_block/private__multi_head_self_attention/dense_2/kernelJprivate__transformer_block/private__multi_head_self_attention/dense_2/biasLprivate__transformer_block/private__multi_head_self_attention/dense_3/kernelJprivate__transformer_block/private__multi_head_self_attention/dense_3/biasdense_4/kerneldense_4/biasdense_5/kerneldense_5/bias4private__transformer_block/layer_normalization/gamma3private__transformer_block/layer_normalization/beta6private__transformer_block/layer_normalization_1/gamma5private__transformer_block/layer_normalization_1/beta	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttrue_positivestrue_negativesfalse_positivesfalse_negativesAdam/dense_6/kernel/mAdam/dense_6/bias/mAdam/dense_7/kernel/mAdam/dense_7/bias/mAAdam/private__token_and_position_embedding/embedding/embeddings/mCAdam/private__token_and_position_embedding/embedding_1/embeddings/mQAdam/private__transformer_block/private__multi_head_self_attention/dense/kernel/mOAdam/private__transformer_block/private__multi_head_self_attention/dense/bias/mSAdam/private__transformer_block/private__multi_head_self_attention/dense_1/kernel/mQAdam/private__transformer_block/private__multi_head_self_attention/dense_1/bias/mSAdam/private__transformer_block/private__multi_head_self_attention/dense_2/kernel/mQAdam/private__transformer_block/private__multi_head_self_attention/dense_2/bias/mSAdam/private__transformer_block/private__multi_head_self_attention/dense_3/kernel/mQAdam/private__transformer_block/private__multi_head_self_attention/dense_3/bias/mAdam/dense_4/kernel/mAdam/dense_4/bias/mAdam/dense_5/kernel/mAdam/dense_5/bias/m;Adam/private__transformer_block/layer_normalization/gamma/m:Adam/private__transformer_block/layer_normalization/beta/m=Adam/private__transformer_block/layer_normalization_1/gamma/m<Adam/private__transformer_block/layer_normalization_1/beta/mAdam/dense_6/kernel/vAdam/dense_6/bias/vAdam/dense_7/kernel/vAdam/dense_7/bias/vAAdam/private__token_and_position_embedding/embedding/embeddings/vCAdam/private__token_and_position_embedding/embedding_1/embeddings/vQAdam/private__transformer_block/private__multi_head_self_attention/dense/kernel/vOAdam/private__transformer_block/private__multi_head_self_attention/dense/bias/vSAdam/private__transformer_block/private__multi_head_self_attention/dense_1/kernel/vQAdam/private__transformer_block/private__multi_head_self_attention/dense_1/bias/vSAdam/private__transformer_block/private__multi_head_self_attention/dense_2/kernel/vQAdam/private__transformer_block/private__multi_head_self_attention/dense_2/bias/vSAdam/private__transformer_block/private__multi_head_self_attention/dense_3/kernel/vQAdam/private__transformer_block/private__multi_head_self_attention/dense_3/bias/vAdam/dense_4/kernel/vAdam/dense_4/bias/vAdam/dense_5/kernel/vAdam/dense_5/bias/v;Adam/private__transformer_block/layer_normalization/gamma/v:Adam/private__transformer_block/layer_normalization/beta/v=Adam/private__transformer_block/layer_normalization_1/gamma/v<Adam/private__transformer_block/layer_normalization_1/beta/v*Y
TinR
P2N*
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
 __inference__traced_restore_8196À$
ê

___inference_private__token_and_position_embedding_layer_call_and_return_conditional_losses_4919
x4
!embedding_1_embedding_lookup_4906:	 2
embedding_embedding_lookup_4912:	Æ 
identity¢embedding/embedding_lookup¢embedding_1/embedding_lookup6
ShapeShapex*
T0*
_output_shapes
:f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskM
range/startConst*
_output_shapes
: *
dtype0*
value	B : M
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :o
rangeRangerange/start:output:0strided_slice:output:0range/delta:output:0*
_output_shapes	
:Õ
embedding_1/embedding_lookupResourceGather!embedding_1_embedding_lookup_4906range:output:0*
Tindices0*4
_class*
(&loc:@embedding_1/embedding_lookup/4906*
_output_shapes
:	 *
dtype0¸
%embedding_1/embedding_lookup/IdentityIdentity%embedding_1/embedding_lookup:output:0*
T0*4
_class*
(&loc:@embedding_1/embedding_lookup/4906*
_output_shapes
:	 
'embedding_1/embedding_lookup/Identity_1Identity.embedding_1/embedding_lookup/Identity:output:0*
T0*
_output_shapes
:	 [
embedding/CastCastx*

DstT0*

SrcT0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà
embedding/embedding_lookupResourceGatherembedding_embedding_lookup_4912embedding/Cast:y:0*
Tindices0*2
_class(
&$loc:@embedding/embedding_lookup/4912*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
dtype0¿
#embedding/embedding_lookup/IdentityIdentity#embedding/embedding_lookup:output:0*
T0*2
_class(
&$loc:@embedding/embedding_lookup/4912*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
%embedding/embedding_lookup/Identity_1Identity,embedding/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¥
addAddV2.embedding/embedding_lookup/Identity_1:output:00embedding_1/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ [
IdentityIdentityadd:z:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
NoOpNoOp^embedding/embedding_lookup^embedding_1/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 28
embedding/embedding_lookupembedding/embedding_lookup2<
embedding_1/embedding_lookupembedding_1/embedding_lookup:K G
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_namex
ñ	
b
C__inference_dropout_2_layer_call_and_return_conditional_losses_7415

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=¦
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
ñ	
b
C__inference_dropout_3_layer_call_and_return_conditional_losses_7462

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=¦
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿo
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

¶
$__inference_model_layer_call_fn_6127

inputs
unknown:	 
	unknown_0:	Æ 
	unknown_1:  
	unknown_2: 
	unknown_3:  
	unknown_4: 
	unknown_5:  
	unknown_6: 
	unknown_7:  
	unknown_8: 
	unknown_9: 

unknown_10: 

unknown_11:  

unknown_12: 

unknown_13:  

unknown_14: 

unknown_15: 

unknown_16: 

unknown_17: 

unknown_18:

unknown_19:

unknown_20:
identity¢StatefulPartitionedCallã
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*8
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_5253o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


D__inference_sequential_layer_call_and_return_conditional_losses_4823

inputs
dense_4_4812:  
dense_4_4814: 
dense_5_4817:  
dense_5_4819: 
identity¢dense_4/StatefulPartitionedCall¢dense_5/StatefulPartitionedCallë
dense_4/StatefulPartitionedCallStatefulPartitionedCallinputsdense_4_4812dense_4_4814*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dense_4_layer_call_and_return_conditional_losses_4720
dense_5/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0dense_5_4817dense_5_4819*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dense_5_layer_call_and_return_conditional_losses_4756|
IdentityIdentity(dense_5/StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
NoOpNoOp ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ : : : : 2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
ª
Ì
)__inference_sequential_layer_call_fn_7508

inputs
unknown:  
	unknown_0: 
	unknown_1:  
	unknown_2: 
identity¢StatefulPartitionedCallø
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_4823t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
õú
«
T__inference_private__transformer_block_layer_call_and_return_conditional_losses_7119

inputs\
Jprivate__multi_head_self_attention_dense_tensordot_readvariableop_resource:  V
Hprivate__multi_head_self_attention_dense_biasadd_readvariableop_resource: ^
Lprivate__multi_head_self_attention_dense_1_tensordot_readvariableop_resource:  X
Jprivate__multi_head_self_attention_dense_1_biasadd_readvariableop_resource: ^
Lprivate__multi_head_self_attention_dense_2_tensordot_readvariableop_resource:  X
Jprivate__multi_head_self_attention_dense_2_biasadd_readvariableop_resource: ^
Lprivate__multi_head_self_attention_dense_3_tensordot_readvariableop_resource:  X
Jprivate__multi_head_self_attention_dense_3_biasadd_readvariableop_resource: G
9layer_normalization_batchnorm_mul_readvariableop_resource: C
5layer_normalization_batchnorm_readvariableop_resource: F
4sequential_dense_4_tensordot_readvariableop_resource:  @
2sequential_dense_4_biasadd_readvariableop_resource: F
4sequential_dense_5_tensordot_readvariableop_resource:  @
2sequential_dense_5_biasadd_readvariableop_resource: I
;layer_normalization_1_batchnorm_mul_readvariableop_resource: E
7layer_normalization_1_batchnorm_readvariableop_resource: 
identity¢,layer_normalization/batchnorm/ReadVariableOp¢0layer_normalization/batchnorm/mul/ReadVariableOp¢.layer_normalization_1/batchnorm/ReadVariableOp¢2layer_normalization_1/batchnorm/mul/ReadVariableOp¢?private__multi_head_self_attention/dense/BiasAdd/ReadVariableOp¢Aprivate__multi_head_self_attention/dense/Tensordot/ReadVariableOp¢Aprivate__multi_head_self_attention/dense_1/BiasAdd/ReadVariableOp¢Cprivate__multi_head_self_attention/dense_1/Tensordot/ReadVariableOp¢Aprivate__multi_head_self_attention/dense_2/BiasAdd/ReadVariableOp¢Cprivate__multi_head_self_attention/dense_2/Tensordot/ReadVariableOp¢Aprivate__multi_head_self_attention/dense_3/BiasAdd/ReadVariableOp¢Cprivate__multi_head_self_attention/dense_3/Tensordot/ReadVariableOp¢)sequential/dense_4/BiasAdd/ReadVariableOp¢+sequential/dense_4/Tensordot/ReadVariableOp¢)sequential/dense_5/BiasAdd/ReadVariableOp¢+sequential/dense_5/Tensordot/ReadVariableOp^
(private__multi_head_self_attention/ShapeShapeinputs*
T0*
_output_shapes
:
6private__multi_head_self_attention/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
8private__multi_head_self_attention/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
8private__multi_head_self_attention/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
0private__multi_head_self_attention/strided_sliceStridedSlice1private__multi_head_self_attention/Shape:output:0?private__multi_head_self_attention/strided_slice/stack:output:0Aprivate__multi_head_self_attention/strided_slice/stack_1:output:0Aprivate__multi_head_self_attention/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskÌ
Aprivate__multi_head_self_attention/dense/Tensordot/ReadVariableOpReadVariableOpJprivate__multi_head_self_attention_dense_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype0
7private__multi_head_self_attention/dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:
7private__multi_head_self_attention/dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       n
8private__multi_head_self_attention/dense/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:
@private__multi_head_self_attention/dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ß
;private__multi_head_self_attention/dense/Tensordot/GatherV2GatherV2Aprivate__multi_head_self_attention/dense/Tensordot/Shape:output:0@private__multi_head_self_attention/dense/Tensordot/free:output:0Iprivate__multi_head_self_attention/dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:
Bprivate__multi_head_self_attention/dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ã
=private__multi_head_self_attention/dense/Tensordot/GatherV2_1GatherV2Aprivate__multi_head_self_attention/dense/Tensordot/Shape:output:0@private__multi_head_self_attention/dense/Tensordot/axes:output:0Kprivate__multi_head_self_attention/dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:
8private__multi_head_self_attention/dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: é
7private__multi_head_self_attention/dense/Tensordot/ProdProdDprivate__multi_head_self_attention/dense/Tensordot/GatherV2:output:0Aprivate__multi_head_self_attention/dense/Tensordot/Const:output:0*
T0*
_output_shapes
: 
:private__multi_head_self_attention/dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ï
9private__multi_head_self_attention/dense/Tensordot/Prod_1ProdFprivate__multi_head_self_attention/dense/Tensordot/GatherV2_1:output:0Cprivate__multi_head_self_attention/dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 
>private__multi_head_self_attention/dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : À
9private__multi_head_self_attention/dense/Tensordot/concatConcatV2@private__multi_head_self_attention/dense/Tensordot/free:output:0@private__multi_head_self_attention/dense/Tensordot/axes:output:0Gprivate__multi_head_self_attention/dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:ô
8private__multi_head_self_attention/dense/Tensordot/stackPack@private__multi_head_self_attention/dense/Tensordot/Prod:output:0Bprivate__multi_head_self_attention/dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:Ì
<private__multi_head_self_attention/dense/Tensordot/transpose	TransposeinputsBprivate__multi_head_self_attention/dense/Tensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
:private__multi_head_self_attention/dense/Tensordot/ReshapeReshape@private__multi_head_self_attention/dense/Tensordot/transpose:y:0Aprivate__multi_head_self_attention/dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
9private__multi_head_self_attention/dense/Tensordot/MatMulMatMulCprivate__multi_head_self_attention/dense/Tensordot/Reshape:output:0Iprivate__multi_head_self_attention/dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
:private__multi_head_self_attention/dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 
@private__multi_head_self_attention/dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ë
;private__multi_head_self_attention/dense/Tensordot/concat_1ConcatV2Dprivate__multi_head_self_attention/dense/Tensordot/GatherV2:output:0Cprivate__multi_head_self_attention/dense/Tensordot/Const_2:output:0Iprivate__multi_head_self_attention/dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:ÿ
2private__multi_head_self_attention/dense/TensordotReshapeCprivate__multi_head_self_attention/dense/Tensordot/MatMul:product:0Dprivate__multi_head_self_attention/dense/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ä
?private__multi_head_self_attention/dense/BiasAdd/ReadVariableOpReadVariableOpHprivate__multi_head_self_attention_dense_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0ø
0private__multi_head_self_attention/dense/BiasAddBiasAdd;private__multi_head_self_attention/dense/Tensordot:output:0Gprivate__multi_head_self_attention/dense/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ð
Cprivate__multi_head_self_attention/dense_1/Tensordot/ReadVariableOpReadVariableOpLprivate__multi_head_self_attention_dense_1_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype0
9private__multi_head_self_attention/dense_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:
9private__multi_head_self_attention/dense_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       p
:private__multi_head_self_attention/dense_1/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:
Bprivate__multi_head_self_attention/dense_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ç
=private__multi_head_self_attention/dense_1/Tensordot/GatherV2GatherV2Cprivate__multi_head_self_attention/dense_1/Tensordot/Shape:output:0Bprivate__multi_head_self_attention/dense_1/Tensordot/free:output:0Kprivate__multi_head_self_attention/dense_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:
Dprivate__multi_head_self_attention/dense_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ë
?private__multi_head_self_attention/dense_1/Tensordot/GatherV2_1GatherV2Cprivate__multi_head_self_attention/dense_1/Tensordot/Shape:output:0Bprivate__multi_head_self_attention/dense_1/Tensordot/axes:output:0Mprivate__multi_head_self_attention/dense_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:
:private__multi_head_self_attention/dense_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ï
9private__multi_head_self_attention/dense_1/Tensordot/ProdProdFprivate__multi_head_self_attention/dense_1/Tensordot/GatherV2:output:0Cprivate__multi_head_self_attention/dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: 
<private__multi_head_self_attention/dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: õ
;private__multi_head_self_attention/dense_1/Tensordot/Prod_1ProdHprivate__multi_head_self_attention/dense_1/Tensordot/GatherV2_1:output:0Eprivate__multi_head_self_attention/dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 
@private__multi_head_self_attention/dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : È
;private__multi_head_self_attention/dense_1/Tensordot/concatConcatV2Bprivate__multi_head_self_attention/dense_1/Tensordot/free:output:0Bprivate__multi_head_self_attention/dense_1/Tensordot/axes:output:0Iprivate__multi_head_self_attention/dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:ú
:private__multi_head_self_attention/dense_1/Tensordot/stackPackBprivate__multi_head_self_attention/dense_1/Tensordot/Prod:output:0Dprivate__multi_head_self_attention/dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:Ð
>private__multi_head_self_attention/dense_1/Tensordot/transpose	TransposeinputsDprivate__multi_head_self_attention/dense_1/Tensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
<private__multi_head_self_attention/dense_1/Tensordot/ReshapeReshapeBprivate__multi_head_self_attention/dense_1/Tensordot/transpose:y:0Cprivate__multi_head_self_attention/dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
;private__multi_head_self_attention/dense_1/Tensordot/MatMulMatMulEprivate__multi_head_self_attention/dense_1/Tensordot/Reshape:output:0Kprivate__multi_head_self_attention/dense_1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
<private__multi_head_self_attention/dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 
Bprivate__multi_head_self_attention/dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ó
=private__multi_head_self_attention/dense_1/Tensordot/concat_1ConcatV2Fprivate__multi_head_self_attention/dense_1/Tensordot/GatherV2:output:0Eprivate__multi_head_self_attention/dense_1/Tensordot/Const_2:output:0Kprivate__multi_head_self_attention/dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
4private__multi_head_self_attention/dense_1/TensordotReshapeEprivate__multi_head_self_attention/dense_1/Tensordot/MatMul:product:0Fprivate__multi_head_self_attention/dense_1/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ È
Aprivate__multi_head_self_attention/dense_1/BiasAdd/ReadVariableOpReadVariableOpJprivate__multi_head_self_attention_dense_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0þ
2private__multi_head_self_attention/dense_1/BiasAddBiasAdd=private__multi_head_self_attention/dense_1/Tensordot:output:0Iprivate__multi_head_self_attention/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ð
Cprivate__multi_head_self_attention/dense_2/Tensordot/ReadVariableOpReadVariableOpLprivate__multi_head_self_attention_dense_2_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype0
9private__multi_head_self_attention/dense_2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:
9private__multi_head_self_attention/dense_2/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       p
:private__multi_head_self_attention/dense_2/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:
Bprivate__multi_head_self_attention/dense_2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ç
=private__multi_head_self_attention/dense_2/Tensordot/GatherV2GatherV2Cprivate__multi_head_self_attention/dense_2/Tensordot/Shape:output:0Bprivate__multi_head_self_attention/dense_2/Tensordot/free:output:0Kprivate__multi_head_self_attention/dense_2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:
Dprivate__multi_head_self_attention/dense_2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ë
?private__multi_head_self_attention/dense_2/Tensordot/GatherV2_1GatherV2Cprivate__multi_head_self_attention/dense_2/Tensordot/Shape:output:0Bprivate__multi_head_self_attention/dense_2/Tensordot/axes:output:0Mprivate__multi_head_self_attention/dense_2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:
:private__multi_head_self_attention/dense_2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ï
9private__multi_head_self_attention/dense_2/Tensordot/ProdProdFprivate__multi_head_self_attention/dense_2/Tensordot/GatherV2:output:0Cprivate__multi_head_self_attention/dense_2/Tensordot/Const:output:0*
T0*
_output_shapes
: 
<private__multi_head_self_attention/dense_2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: õ
;private__multi_head_self_attention/dense_2/Tensordot/Prod_1ProdHprivate__multi_head_self_attention/dense_2/Tensordot/GatherV2_1:output:0Eprivate__multi_head_self_attention/dense_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 
@private__multi_head_self_attention/dense_2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : È
;private__multi_head_self_attention/dense_2/Tensordot/concatConcatV2Bprivate__multi_head_self_attention/dense_2/Tensordot/free:output:0Bprivate__multi_head_self_attention/dense_2/Tensordot/axes:output:0Iprivate__multi_head_self_attention/dense_2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:ú
:private__multi_head_self_attention/dense_2/Tensordot/stackPackBprivate__multi_head_self_attention/dense_2/Tensordot/Prod:output:0Dprivate__multi_head_self_attention/dense_2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:Ð
>private__multi_head_self_attention/dense_2/Tensordot/transpose	TransposeinputsDprivate__multi_head_self_attention/dense_2/Tensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
<private__multi_head_self_attention/dense_2/Tensordot/ReshapeReshapeBprivate__multi_head_self_attention/dense_2/Tensordot/transpose:y:0Cprivate__multi_head_self_attention/dense_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
;private__multi_head_self_attention/dense_2/Tensordot/MatMulMatMulEprivate__multi_head_self_attention/dense_2/Tensordot/Reshape:output:0Kprivate__multi_head_self_attention/dense_2/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
<private__multi_head_self_attention/dense_2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 
Bprivate__multi_head_self_attention/dense_2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ó
=private__multi_head_self_attention/dense_2/Tensordot/concat_1ConcatV2Fprivate__multi_head_self_attention/dense_2/Tensordot/GatherV2:output:0Eprivate__multi_head_self_attention/dense_2/Tensordot/Const_2:output:0Kprivate__multi_head_self_attention/dense_2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
4private__multi_head_self_attention/dense_2/TensordotReshapeEprivate__multi_head_self_attention/dense_2/Tensordot/MatMul:product:0Fprivate__multi_head_self_attention/dense_2/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ È
Aprivate__multi_head_self_attention/dense_2/BiasAdd/ReadVariableOpReadVariableOpJprivate__multi_head_self_attention_dense_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0þ
2private__multi_head_self_attention/dense_2/BiasAddBiasAdd=private__multi_head_self_attention/dense_2/Tensordot:output:0Iprivate__multi_head_self_attention/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ }
2private__multi_head_self_attention/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿt
2private__multi_head_self_attention/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :t
2private__multi_head_self_attention/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :Ø
0private__multi_head_self_attention/Reshape/shapePack9private__multi_head_self_attention/strided_slice:output:0;private__multi_head_self_attention/Reshape/shape/1:output:0;private__multi_head_self_attention/Reshape/shape/2:output:0;private__multi_head_self_attention/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:î
*private__multi_head_self_attention/ReshapeReshape9private__multi_head_self_attention/dense/BiasAdd:output:09private__multi_head_self_attention/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
1private__multi_head_self_attention/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             í
,private__multi_head_self_attention/transpose	Transpose3private__multi_head_self_attention/Reshape:output:0:private__multi_head_self_attention/transpose/perm:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
4private__multi_head_self_attention/Reshape_1/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿv
4private__multi_head_self_attention/Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :v
4private__multi_head_self_attention/Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :à
2private__multi_head_self_attention/Reshape_1/shapePack9private__multi_head_self_attention/strided_slice:output:0=private__multi_head_self_attention/Reshape_1/shape/1:output:0=private__multi_head_self_attention/Reshape_1/shape/2:output:0=private__multi_head_self_attention/Reshape_1/shape/3:output:0*
N*
T0*
_output_shapes
:ô
,private__multi_head_self_attention/Reshape_1Reshape;private__multi_head_self_attention/dense_1/BiasAdd:output:0;private__multi_head_self_attention/Reshape_1/shape:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
3private__multi_head_self_attention/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             ó
.private__multi_head_self_attention/transpose_1	Transpose5private__multi_head_self_attention/Reshape_1:output:0<private__multi_head_self_attention/transpose_1/perm:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
4private__multi_head_self_attention/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿv
4private__multi_head_self_attention/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :v
4private__multi_head_self_attention/Reshape_2/shape/3Const*
_output_shapes
: *
dtype0*
value	B :à
2private__multi_head_self_attention/Reshape_2/shapePack9private__multi_head_self_attention/strided_slice:output:0=private__multi_head_self_attention/Reshape_2/shape/1:output:0=private__multi_head_self_attention/Reshape_2/shape/2:output:0=private__multi_head_self_attention/Reshape_2/shape/3:output:0*
N*
T0*
_output_shapes
:ô
,private__multi_head_self_attention/Reshape_2Reshape;private__multi_head_self_attention/dense_2/BiasAdd:output:0;private__multi_head_self_attention/Reshape_2/shape:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
3private__multi_head_self_attention/transpose_2/permConst*
_output_shapes
:*
dtype0*%
valueB"             ó
.private__multi_head_self_attention/transpose_2	Transpose5private__multi_head_self_attention/Reshape_2:output:0<private__multi_head_self_attention/transpose_2/perm:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿù
)private__multi_head_self_attention/MatMulBatchMatMulV20private__multi_head_self_attention/transpose:y:02private__multi_head_self_attention/transpose_1:y:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
adj_y(
*private__multi_head_self_attention/Shape_1Shape2private__multi_head_self_attention/transpose_1:y:0*
T0*
_output_shapes
:
8private__multi_head_self_attention/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ
:private__multi_head_self_attention/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
:private__multi_head_self_attention/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
2private__multi_head_self_attention/strided_slice_1StridedSlice3private__multi_head_self_attention/Shape_1:output:0Aprivate__multi_head_self_attention/strided_slice_1/stack:output:0Cprivate__multi_head_self_attention/strided_slice_1/stack_1:output:0Cprivate__multi_head_self_attention/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
'private__multi_head_self_attention/CastCast;private__multi_head_self_attention/strided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: }
'private__multi_head_self_attention/SqrtSqrt+private__multi_head_self_attention/Cast:y:0*
T0*
_output_shapes
: â
*private__multi_head_self_attention/truedivRealDiv2private__multi_head_self_attention/MatMul:output:0+private__multi_head_self_attention/Sqrt:y:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ±
*private__multi_head_self_attention/SoftmaxSoftmax.private__multi_head_self_attention/truediv:z:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿé
+private__multi_head_self_attention/MatMul_1BatchMatMulV24private__multi_head_self_attention/Softmax:softmax:02private__multi_head_self_attention/transpose_2:y:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
3private__multi_head_self_attention/transpose_3/permConst*
_output_shapes
:*
dtype0*%
valueB"             ò
.private__multi_head_self_attention/transpose_3	Transpose4private__multi_head_self_attention/MatMul_1:output:0<private__multi_head_self_attention/transpose_3/perm:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
4private__multi_head_self_attention/Reshape_3/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿv
4private__multi_head_self_attention/Reshape_3/shape/2Const*
_output_shapes
: *
dtype0*
value	B : ¡
2private__multi_head_self_attention/Reshape_3/shapePack9private__multi_head_self_attention/strided_slice:output:0=private__multi_head_self_attention/Reshape_3/shape/1:output:0=private__multi_head_self_attention/Reshape_3/shape/2:output:0*
N*
T0*
_output_shapes
:ç
,private__multi_head_self_attention/Reshape_3Reshape2private__multi_head_self_attention/transpose_3:y:0;private__multi_head_self_attention/Reshape_3/shape:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ Ð
Cprivate__multi_head_self_attention/dense_3/Tensordot/ReadVariableOpReadVariableOpLprivate__multi_head_self_attention_dense_3_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype0
9private__multi_head_self_attention/dense_3/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:
9private__multi_head_self_attention/dense_3/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       
:private__multi_head_self_attention/dense_3/Tensordot/ShapeShape5private__multi_head_self_attention/Reshape_3:output:0*
T0*
_output_shapes
:
Bprivate__multi_head_self_attention/dense_3/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ç
=private__multi_head_self_attention/dense_3/Tensordot/GatherV2GatherV2Cprivate__multi_head_self_attention/dense_3/Tensordot/Shape:output:0Bprivate__multi_head_self_attention/dense_3/Tensordot/free:output:0Kprivate__multi_head_self_attention/dense_3/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:
Dprivate__multi_head_self_attention/dense_3/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ë
?private__multi_head_self_attention/dense_3/Tensordot/GatherV2_1GatherV2Cprivate__multi_head_self_attention/dense_3/Tensordot/Shape:output:0Bprivate__multi_head_self_attention/dense_3/Tensordot/axes:output:0Mprivate__multi_head_self_attention/dense_3/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:
:private__multi_head_self_attention/dense_3/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ï
9private__multi_head_self_attention/dense_3/Tensordot/ProdProdFprivate__multi_head_self_attention/dense_3/Tensordot/GatherV2:output:0Cprivate__multi_head_self_attention/dense_3/Tensordot/Const:output:0*
T0*
_output_shapes
: 
<private__multi_head_self_attention/dense_3/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: õ
;private__multi_head_self_attention/dense_3/Tensordot/Prod_1ProdHprivate__multi_head_self_attention/dense_3/Tensordot/GatherV2_1:output:0Eprivate__multi_head_self_attention/dense_3/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 
@private__multi_head_self_attention/dense_3/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : È
;private__multi_head_self_attention/dense_3/Tensordot/concatConcatV2Bprivate__multi_head_self_attention/dense_3/Tensordot/free:output:0Bprivate__multi_head_self_attention/dense_3/Tensordot/axes:output:0Iprivate__multi_head_self_attention/dense_3/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:ú
:private__multi_head_self_attention/dense_3/Tensordot/stackPackBprivate__multi_head_self_attention/dense_3/Tensordot/Prod:output:0Dprivate__multi_head_self_attention/dense_3/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:
>private__multi_head_self_attention/dense_3/Tensordot/transpose	Transpose5private__multi_head_self_attention/Reshape_3:output:0Dprivate__multi_head_self_attention/dense_3/Tensordot/concat:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
<private__multi_head_self_attention/dense_3/Tensordot/ReshapeReshapeBprivate__multi_head_self_attention/dense_3/Tensordot/transpose:y:0Cprivate__multi_head_self_attention/dense_3/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
;private__multi_head_self_attention/dense_3/Tensordot/MatMulMatMulEprivate__multi_head_self_attention/dense_3/Tensordot/Reshape:output:0Kprivate__multi_head_self_attention/dense_3/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
<private__multi_head_self_attention/dense_3/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 
Bprivate__multi_head_self_attention/dense_3/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ó
=private__multi_head_self_attention/dense_3/Tensordot/concat_1ConcatV2Fprivate__multi_head_self_attention/dense_3/Tensordot/GatherV2:output:0Eprivate__multi_head_self_attention/dense_3/Tensordot/Const_2:output:0Kprivate__multi_head_self_attention/dense_3/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
4private__multi_head_self_attention/dense_3/TensordotReshapeEprivate__multi_head_self_attention/dense_3/Tensordot/MatMul:product:0Fprivate__multi_head_self_attention/dense_3/Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ È
Aprivate__multi_head_self_attention/dense_3/BiasAdd/ReadVariableOpReadVariableOpJprivate__multi_head_self_attention_dense_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
2private__multi_head_self_attention/dense_3/BiasAddBiasAdd=private__multi_head_self_attention/dense_3/Tensordot:output:0Iprivate__multi_head_self_attention/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
dropout/IdentityIdentity;private__multi_head_self_attention/dense_3/BiasAdd:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ f
addAddV2inputsdropout/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ |
2layer_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:¶
 layer_normalization/moments/meanMeanadd:z:0;layer_normalization/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(
(layer_normalization/moments/StopGradientStopGradient)layer_normalization/moments/mean:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿµ
-layer_normalization/moments/SquaredDifferenceSquaredDifferenceadd:z:01layer_normalization/moments/StopGradient:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
6layer_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:è
$layer_normalization/moments/varianceMean1layer_normalization/moments/SquaredDifference:z:0?layer_normalization/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(h
#layer_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½75¾
!layer_normalization/batchnorm/addAddV2-layer_normalization/moments/variance:output:0,layer_normalization/batchnorm/add/y:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#layer_normalization/batchnorm/RsqrtRsqrt%layer_normalization/batchnorm/add:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¦
0layer_normalization/batchnorm/mul/ReadVariableOpReadVariableOp9layer_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype0Â
!layer_normalization/batchnorm/mulMul'layer_normalization/batchnorm/Rsqrt:y:08layer_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
#layer_normalization/batchnorm/mul_1Muladd:z:0%layer_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ³
#layer_normalization/batchnorm/mul_2Mul)layer_normalization/moments/mean:output:0%layer_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
,layer_normalization/batchnorm/ReadVariableOpReadVariableOp5layer_normalization_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype0¾
!layer_normalization/batchnorm/subSub4layer_normalization/batchnorm/ReadVariableOp:value:0'layer_normalization/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ³
#layer_normalization/batchnorm/add_1AddV2'layer_normalization/batchnorm/mul_1:z:0%layer_normalization/batchnorm/sub:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
+sequential/dense_4/Tensordot/ReadVariableOpReadVariableOp4sequential_dense_4_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype0k
!sequential/dense_4/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:r
!sequential/dense_4/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       y
"sequential/dense_4/Tensordot/ShapeShape'layer_normalization/batchnorm/add_1:z:0*
T0*
_output_shapes
:l
*sequential/dense_4/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
%sequential/dense_4/Tensordot/GatherV2GatherV2+sequential/dense_4/Tensordot/Shape:output:0*sequential/dense_4/Tensordot/free:output:03sequential/dense_4/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:n
,sequential/dense_4/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
'sequential/dense_4/Tensordot/GatherV2_1GatherV2+sequential/dense_4/Tensordot/Shape:output:0*sequential/dense_4/Tensordot/axes:output:05sequential/dense_4/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:l
"sequential/dense_4/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: §
!sequential/dense_4/Tensordot/ProdProd.sequential/dense_4/Tensordot/GatherV2:output:0+sequential/dense_4/Tensordot/Const:output:0*
T0*
_output_shapes
: n
$sequential/dense_4/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ­
#sequential/dense_4/Tensordot/Prod_1Prod0sequential/dense_4/Tensordot/GatherV2_1:output:0-sequential/dense_4/Tensordot/Const_1:output:0*
T0*
_output_shapes
: j
(sequential/dense_4/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : è
#sequential/dense_4/Tensordot/concatConcatV2*sequential/dense_4/Tensordot/free:output:0*sequential/dense_4/Tensordot/axes:output:01sequential/dense_4/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:²
"sequential/dense_4/Tensordot/stackPack*sequential/dense_4/Tensordot/Prod:output:0,sequential/dense_4/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:Á
&sequential/dense_4/Tensordot/transpose	Transpose'layer_normalization/batchnorm/add_1:z:0,sequential/dense_4/Tensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ã
$sequential/dense_4/Tensordot/ReshapeReshape*sequential/dense_4/Tensordot/transpose:y:0+sequential/dense_4/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÃ
#sequential/dense_4/Tensordot/MatMulMatMul-sequential/dense_4/Tensordot/Reshape:output:03sequential/dense_4/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ n
$sequential/dense_4/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: l
*sequential/dense_4/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ó
%sequential/dense_4/Tensordot/concat_1ConcatV2.sequential/dense_4/Tensordot/GatherV2:output:0-sequential/dense_4/Tensordot/Const_2:output:03sequential/dense_4/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:½
sequential/dense_4/TensordotReshape-sequential/dense_4/Tensordot/MatMul:product:0.sequential/dense_4/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
)sequential/dense_4/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0¶
sequential/dense_4/BiasAddBiasAdd%sequential/dense_4/Tensordot:output:01sequential/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ {
sequential/dense_4/ReluRelu#sequential/dense_4/BiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
+sequential/dense_5/Tensordot/ReadVariableOpReadVariableOp4sequential_dense_5_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype0k
!sequential/dense_5/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:r
!sequential/dense_5/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       w
"sequential/dense_5/Tensordot/ShapeShape%sequential/dense_4/Relu:activations:0*
T0*
_output_shapes
:l
*sequential/dense_5/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
%sequential/dense_5/Tensordot/GatherV2GatherV2+sequential/dense_5/Tensordot/Shape:output:0*sequential/dense_5/Tensordot/free:output:03sequential/dense_5/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:n
,sequential/dense_5/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
'sequential/dense_5/Tensordot/GatherV2_1GatherV2+sequential/dense_5/Tensordot/Shape:output:0*sequential/dense_5/Tensordot/axes:output:05sequential/dense_5/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:l
"sequential/dense_5/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: §
!sequential/dense_5/Tensordot/ProdProd.sequential/dense_5/Tensordot/GatherV2:output:0+sequential/dense_5/Tensordot/Const:output:0*
T0*
_output_shapes
: n
$sequential/dense_5/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ­
#sequential/dense_5/Tensordot/Prod_1Prod0sequential/dense_5/Tensordot/GatherV2_1:output:0-sequential/dense_5/Tensordot/Const_1:output:0*
T0*
_output_shapes
: j
(sequential/dense_5/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : è
#sequential/dense_5/Tensordot/concatConcatV2*sequential/dense_5/Tensordot/free:output:0*sequential/dense_5/Tensordot/axes:output:01sequential/dense_5/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:²
"sequential/dense_5/Tensordot/stackPack*sequential/dense_5/Tensordot/Prod:output:0,sequential/dense_5/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:¿
&sequential/dense_5/Tensordot/transpose	Transpose%sequential/dense_4/Relu:activations:0,sequential/dense_5/Tensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ã
$sequential/dense_5/Tensordot/ReshapeReshape*sequential/dense_5/Tensordot/transpose:y:0+sequential/dense_5/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÃ
#sequential/dense_5/Tensordot/MatMulMatMul-sequential/dense_5/Tensordot/Reshape:output:03sequential/dense_5/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ n
$sequential/dense_5/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: l
*sequential/dense_5/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ó
%sequential/dense_5/Tensordot/concat_1ConcatV2.sequential/dense_5/Tensordot/GatherV2:output:0-sequential/dense_5/Tensordot/Const_2:output:03sequential/dense_5/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:½
sequential/dense_5/TensordotReshape-sequential/dense_5/Tensordot/MatMul:product:0.sequential/dense_5/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
)sequential/dense_5/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0¶
sequential/dense_5/BiasAddBiasAdd%sequential/dense_5/Tensordot:output:01sequential/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ z
dropout_1/IdentityIdentity#sequential/dense_5/BiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
add_1AddV2'layer_normalization/batchnorm/add_1:z:0dropout_1/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ~
4layer_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:¼
"layer_normalization_1/moments/meanMean	add_1:z:0=layer_normalization_1/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(
*layer_normalization_1/moments/StopGradientStopGradient+layer_normalization_1/moments/mean:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ»
/layer_normalization_1/moments/SquaredDifferenceSquaredDifference	add_1:z:03layer_normalization_1/moments/StopGradient:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
8layer_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:î
&layer_normalization_1/moments/varianceMean3layer_normalization_1/moments/SquaredDifference:z:0Alayer_normalization_1/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(j
%layer_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½75Ä
#layer_normalization_1/batchnorm/addAddV2/layer_normalization_1/moments/variance:output:0.layer_normalization_1/batchnorm/add/y:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%layer_normalization_1/batchnorm/RsqrtRsqrt'layer_normalization_1/batchnorm/add:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
2layer_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype0È
#layer_normalization_1/batchnorm/mulMul)layer_normalization_1/batchnorm/Rsqrt:y:0:layer_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
%layer_normalization_1/batchnorm/mul_1Mul	add_1:z:0'layer_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¹
%layer_normalization_1/batchnorm/mul_2Mul+layer_normalization_1/moments/mean:output:0'layer_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¢
.layer_normalization_1/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype0Ä
#layer_normalization_1/batchnorm/subSub6layer_normalization_1/batchnorm/ReadVariableOp:value:0)layer_normalization_1/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¹
%layer_normalization_1/batchnorm/add_1AddV2)layer_normalization_1/batchnorm/mul_1:z:0'layer_normalization_1/batchnorm/sub:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ }
IdentityIdentity)layer_normalization_1/batchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ æ
NoOpNoOp-^layer_normalization/batchnorm/ReadVariableOp1^layer_normalization/batchnorm/mul/ReadVariableOp/^layer_normalization_1/batchnorm/ReadVariableOp3^layer_normalization_1/batchnorm/mul/ReadVariableOp@^private__multi_head_self_attention/dense/BiasAdd/ReadVariableOpB^private__multi_head_self_attention/dense/Tensordot/ReadVariableOpB^private__multi_head_self_attention/dense_1/BiasAdd/ReadVariableOpD^private__multi_head_self_attention/dense_1/Tensordot/ReadVariableOpB^private__multi_head_self_attention/dense_2/BiasAdd/ReadVariableOpD^private__multi_head_self_attention/dense_2/Tensordot/ReadVariableOpB^private__multi_head_self_attention/dense_3/BiasAdd/ReadVariableOpD^private__multi_head_self_attention/dense_3/Tensordot/ReadVariableOp*^sequential/dense_4/BiasAdd/ReadVariableOp,^sequential/dense_4/Tensordot/ReadVariableOp*^sequential/dense_5/BiasAdd/ReadVariableOp,^sequential/dense_5/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:ÿÿÿÿÿÿÿÿÿ : : : : : : : : : : : : : : : : 2\
,layer_normalization/batchnorm/ReadVariableOp,layer_normalization/batchnorm/ReadVariableOp2d
0layer_normalization/batchnorm/mul/ReadVariableOp0layer_normalization/batchnorm/mul/ReadVariableOp2`
.layer_normalization_1/batchnorm/ReadVariableOp.layer_normalization_1/batchnorm/ReadVariableOp2h
2layer_normalization_1/batchnorm/mul/ReadVariableOp2layer_normalization_1/batchnorm/mul/ReadVariableOp2
?private__multi_head_self_attention/dense/BiasAdd/ReadVariableOp?private__multi_head_self_attention/dense/BiasAdd/ReadVariableOp2
Aprivate__multi_head_self_attention/dense/Tensordot/ReadVariableOpAprivate__multi_head_self_attention/dense/Tensordot/ReadVariableOp2
Aprivate__multi_head_self_attention/dense_1/BiasAdd/ReadVariableOpAprivate__multi_head_self_attention/dense_1/BiasAdd/ReadVariableOp2
Cprivate__multi_head_self_attention/dense_1/Tensordot/ReadVariableOpCprivate__multi_head_self_attention/dense_1/Tensordot/ReadVariableOp2
Aprivate__multi_head_self_attention/dense_2/BiasAdd/ReadVariableOpAprivate__multi_head_self_attention/dense_2/BiasAdd/ReadVariableOp2
Cprivate__multi_head_self_attention/dense_2/Tensordot/ReadVariableOpCprivate__multi_head_self_attention/dense_2/Tensordot/ReadVariableOp2
Aprivate__multi_head_self_attention/dense_3/BiasAdd/ReadVariableOpAprivate__multi_head_self_attention/dense_3/BiasAdd/ReadVariableOp2
Cprivate__multi_head_self_attention/dense_3/Tensordot/ReadVariableOpCprivate__multi_head_self_attention/dense_3/Tensordot/ReadVariableOp2V
)sequential/dense_4/BiasAdd/ReadVariableOp)sequential/dense_4/BiasAdd/ReadVariableOp2Z
+sequential/dense_4/Tensordot/ReadVariableOp+sequential/dense_4/Tensordot/ReadVariableOp2V
)sequential/dense_5/BiasAdd/ReadVariableOp)sequential/dense_5/BiasAdd/ReadVariableOp2Z
+sequential/dense_5/Tensordot/ReadVariableOp+sequential/dense_5/Tensordot/ReadVariableOp:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs

·
$__inference_model_layer_call_fn_5300
input_1
unknown:	 
	unknown_0:	Æ 
	unknown_1:  
	unknown_2: 
	unknown_3:  
	unknown_4: 
	unknown_5:  
	unknown_6: 
	unknown_7:  
	unknown_8: 
	unknown_9: 

unknown_10: 

unknown_11:  

unknown_12: 

unknown_13:  

unknown_14: 

unknown_15: 

unknown_16: 

unknown_17: 

unknown_18:

unknown_19:

unknown_20:
identity¢StatefulPartitionedCallä
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
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*8
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_5253o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
Ì
Æ?
 __inference__traced_restore_8196
file_prefix1
assignvariableop_dense_6_kernel: -
assignvariableop_1_dense_6_bias:3
!assignvariableop_2_dense_7_kernel:-
assignvariableop_3_dense_7_bias:`
Massignvariableop_4_private__token_and_position_embedding_embedding_embeddings:	Æ b
Oassignvariableop_5_private__token_and_position_embedding_embedding_1_embeddings:	 o
]assignvariableop_6_private__transformer_block_private__multi_head_self_attention_dense_kernel:  i
[assignvariableop_7_private__transformer_block_private__multi_head_self_attention_dense_bias: q
_assignvariableop_8_private__transformer_block_private__multi_head_self_attention_dense_1_kernel:  k
]assignvariableop_9_private__transformer_block_private__multi_head_self_attention_dense_1_bias: r
`assignvariableop_10_private__transformer_block_private__multi_head_self_attention_dense_2_kernel:  l
^assignvariableop_11_private__transformer_block_private__multi_head_self_attention_dense_2_bias: r
`assignvariableop_12_private__transformer_block_private__multi_head_self_attention_dense_3_kernel:  l
^assignvariableop_13_private__transformer_block_private__multi_head_self_attention_dense_3_bias: 4
"assignvariableop_14_dense_4_kernel:  .
 assignvariableop_15_dense_4_bias: 4
"assignvariableop_16_dense_5_kernel:  .
 assignvariableop_17_dense_5_bias: V
Hassignvariableop_18_private__transformer_block_layer_normalization_gamma: U
Gassignvariableop_19_private__transformer_block_layer_normalization_beta: X
Jassignvariableop_20_private__transformer_block_layer_normalization_1_gamma: W
Iassignvariableop_21_private__transformer_block_layer_normalization_1_beta: '
assignvariableop_22_adam_iter:	 )
assignvariableop_23_adam_beta_1: )
assignvariableop_24_adam_beta_2: (
assignvariableop_25_adam_decay: 0
&assignvariableop_26_adam_learning_rate: #
assignvariableop_27_total: #
assignvariableop_28_count: 1
"assignvariableop_29_true_positives:	È1
"assignvariableop_30_true_negatives:	È2
#assignvariableop_31_false_positives:	È2
#assignvariableop_32_false_negatives:	È;
)assignvariableop_33_adam_dense_6_kernel_m: 5
'assignvariableop_34_adam_dense_6_bias_m:;
)assignvariableop_35_adam_dense_7_kernel_m:5
'assignvariableop_36_adam_dense_7_bias_m:h
Uassignvariableop_37_adam_private__token_and_position_embedding_embedding_embeddings_m:	Æ j
Wassignvariableop_38_adam_private__token_and_position_embedding_embedding_1_embeddings_m:	 w
eassignvariableop_39_adam_private__transformer_block_private__multi_head_self_attention_dense_kernel_m:  q
cassignvariableop_40_adam_private__transformer_block_private__multi_head_self_attention_dense_bias_m: y
gassignvariableop_41_adam_private__transformer_block_private__multi_head_self_attention_dense_1_kernel_m:  s
eassignvariableop_42_adam_private__transformer_block_private__multi_head_self_attention_dense_1_bias_m: y
gassignvariableop_43_adam_private__transformer_block_private__multi_head_self_attention_dense_2_kernel_m:  s
eassignvariableop_44_adam_private__transformer_block_private__multi_head_self_attention_dense_2_bias_m: y
gassignvariableop_45_adam_private__transformer_block_private__multi_head_self_attention_dense_3_kernel_m:  s
eassignvariableop_46_adam_private__transformer_block_private__multi_head_self_attention_dense_3_bias_m: ;
)assignvariableop_47_adam_dense_4_kernel_m:  5
'assignvariableop_48_adam_dense_4_bias_m: ;
)assignvariableop_49_adam_dense_5_kernel_m:  5
'assignvariableop_50_adam_dense_5_bias_m: ]
Oassignvariableop_51_adam_private__transformer_block_layer_normalization_gamma_m: \
Nassignvariableop_52_adam_private__transformer_block_layer_normalization_beta_m: _
Qassignvariableop_53_adam_private__transformer_block_layer_normalization_1_gamma_m: ^
Passignvariableop_54_adam_private__transformer_block_layer_normalization_1_beta_m: ;
)assignvariableop_55_adam_dense_6_kernel_v: 5
'assignvariableop_56_adam_dense_6_bias_v:;
)assignvariableop_57_adam_dense_7_kernel_v:5
'assignvariableop_58_adam_dense_7_bias_v:h
Uassignvariableop_59_adam_private__token_and_position_embedding_embedding_embeddings_v:	Æ j
Wassignvariableop_60_adam_private__token_and_position_embedding_embedding_1_embeddings_v:	 w
eassignvariableop_61_adam_private__transformer_block_private__multi_head_self_attention_dense_kernel_v:  q
cassignvariableop_62_adam_private__transformer_block_private__multi_head_self_attention_dense_bias_v: y
gassignvariableop_63_adam_private__transformer_block_private__multi_head_self_attention_dense_1_kernel_v:  s
eassignvariableop_64_adam_private__transformer_block_private__multi_head_self_attention_dense_1_bias_v: y
gassignvariableop_65_adam_private__transformer_block_private__multi_head_self_attention_dense_2_kernel_v:  s
eassignvariableop_66_adam_private__transformer_block_private__multi_head_self_attention_dense_2_bias_v: y
gassignvariableop_67_adam_private__transformer_block_private__multi_head_self_attention_dense_3_kernel_v:  s
eassignvariableop_68_adam_private__transformer_block_private__multi_head_self_attention_dense_3_bias_v: ;
)assignvariableop_69_adam_dense_4_kernel_v:  5
'assignvariableop_70_adam_dense_4_bias_v: ;
)assignvariableop_71_adam_dense_5_kernel_v:  5
'assignvariableop_72_adam_dense_5_bias_v: ]
Oassignvariableop_73_adam_private__transformer_block_layer_normalization_gamma_v: \
Nassignvariableop_74_adam_private__transformer_block_layer_normalization_beta_v: _
Qassignvariableop_75_adam_private__transformer_block_layer_normalization_1_gamma_v: ^
Passignvariableop_76_adam_private__transformer_block_layer_normalization_1_beta_v: 
identity_78¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_36¢AssignVariableOp_37¢AssignVariableOp_38¢AssignVariableOp_39¢AssignVariableOp_4¢AssignVariableOp_40¢AssignVariableOp_41¢AssignVariableOp_42¢AssignVariableOp_43¢AssignVariableOp_44¢AssignVariableOp_45¢AssignVariableOp_46¢AssignVariableOp_47¢AssignVariableOp_48¢AssignVariableOp_49¢AssignVariableOp_5¢AssignVariableOp_50¢AssignVariableOp_51¢AssignVariableOp_52¢AssignVariableOp_53¢AssignVariableOp_54¢AssignVariableOp_55¢AssignVariableOp_56¢AssignVariableOp_57¢AssignVariableOp_58¢AssignVariableOp_59¢AssignVariableOp_6¢AssignVariableOp_60¢AssignVariableOp_61¢AssignVariableOp_62¢AssignVariableOp_63¢AssignVariableOp_64¢AssignVariableOp_65¢AssignVariableOp_66¢AssignVariableOp_67¢AssignVariableOp_68¢AssignVariableOp_69¢AssignVariableOp_7¢AssignVariableOp_70¢AssignVariableOp_71¢AssignVariableOp_72¢AssignVariableOp_73¢AssignVariableOp_74¢AssignVariableOp_75¢AssignVariableOp_76¢AssignVariableOp_8¢AssignVariableOp_9È%
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:N*
dtype0*î$
valueä$Bá$NB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/1/true_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/1/true_negatives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/1/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/1/false_negatives/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:N*
dtype0*±
value§B¤NB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B §
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*Î
_output_shapes»
¸::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*\
dtypesR
P2N	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOpassignvariableop_dense_6_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_6_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOp!assignvariableop_2_dense_7_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOpassignvariableop_3_dense_7_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:¼
AssignVariableOp_4AssignVariableOpMassignvariableop_4_private__token_and_position_embedding_embedding_embeddingsIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:¾
AssignVariableOp_5AssignVariableOpOassignvariableop_5_private__token_and_position_embedding_embedding_1_embeddingsIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:Ì
AssignVariableOp_6AssignVariableOp]assignvariableop_6_private__transformer_block_private__multi_head_self_attention_dense_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:Ê
AssignVariableOp_7AssignVariableOp[assignvariableop_7_private__transformer_block_private__multi_head_self_attention_dense_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:Î
AssignVariableOp_8AssignVariableOp_assignvariableop_8_private__transformer_block_private__multi_head_self_attention_dense_1_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:Ì
AssignVariableOp_9AssignVariableOp]assignvariableop_9_private__transformer_block_private__multi_head_self_attention_dense_1_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:Ñ
AssignVariableOp_10AssignVariableOp`assignvariableop_10_private__transformer_block_private__multi_head_self_attention_dense_2_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:Ï
AssignVariableOp_11AssignVariableOp^assignvariableop_11_private__transformer_block_private__multi_head_self_attention_dense_2_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:Ñ
AssignVariableOp_12AssignVariableOp`assignvariableop_12_private__transformer_block_private__multi_head_self_attention_dense_3_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:Ï
AssignVariableOp_13AssignVariableOp^assignvariableop_13_private__transformer_block_private__multi_head_self_attention_dense_3_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_14AssignVariableOp"assignvariableop_14_dense_4_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_15AssignVariableOp assignvariableop_15_dense_4_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_16AssignVariableOp"assignvariableop_16_dense_5_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_17AssignVariableOp assignvariableop_17_dense_5_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:¹
AssignVariableOp_18AssignVariableOpHassignvariableop_18_private__transformer_block_layer_normalization_gammaIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:¸
AssignVariableOp_19AssignVariableOpGassignvariableop_19_private__transformer_block_layer_normalization_betaIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:»
AssignVariableOp_20AssignVariableOpJassignvariableop_20_private__transformer_block_layer_normalization_1_gammaIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:º
AssignVariableOp_21AssignVariableOpIassignvariableop_21_private__transformer_block_layer_normalization_1_betaIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_22AssignVariableOpassignvariableop_22_adam_iterIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_23AssignVariableOpassignvariableop_23_adam_beta_1Identity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_24AssignVariableOpassignvariableop_24_adam_beta_2Identity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_25AssignVariableOpassignvariableop_25_adam_decayIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_26AssignVariableOp&assignvariableop_26_adam_learning_rateIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_27AssignVariableOpassignvariableop_27_totalIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_28AssignVariableOpassignvariableop_28_countIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_29AssignVariableOp"assignvariableop_29_true_positivesIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_30AssignVariableOp"assignvariableop_30_true_negativesIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_31AssignVariableOp#assignvariableop_31_false_positivesIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_32AssignVariableOp#assignvariableop_32_false_negativesIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_33AssignVariableOp)assignvariableop_33_adam_dense_6_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_34AssignVariableOp'assignvariableop_34_adam_dense_6_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_35AssignVariableOp)assignvariableop_35_adam_dense_7_kernel_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_36AssignVariableOp'assignvariableop_36_adam_dense_7_bias_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:Æ
AssignVariableOp_37AssignVariableOpUassignvariableop_37_adam_private__token_and_position_embedding_embedding_embeddings_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:È
AssignVariableOp_38AssignVariableOpWassignvariableop_38_adam_private__token_and_position_embedding_embedding_1_embeddings_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:Ö
AssignVariableOp_39AssignVariableOpeassignvariableop_39_adam_private__transformer_block_private__multi_head_self_attention_dense_kernel_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:Ô
AssignVariableOp_40AssignVariableOpcassignvariableop_40_adam_private__transformer_block_private__multi_head_self_attention_dense_bias_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:Ø
AssignVariableOp_41AssignVariableOpgassignvariableop_41_adam_private__transformer_block_private__multi_head_self_attention_dense_1_kernel_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:Ö
AssignVariableOp_42AssignVariableOpeassignvariableop_42_adam_private__transformer_block_private__multi_head_self_attention_dense_1_bias_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:Ø
AssignVariableOp_43AssignVariableOpgassignvariableop_43_adam_private__transformer_block_private__multi_head_self_attention_dense_2_kernel_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:Ö
AssignVariableOp_44AssignVariableOpeassignvariableop_44_adam_private__transformer_block_private__multi_head_self_attention_dense_2_bias_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:Ø
AssignVariableOp_45AssignVariableOpgassignvariableop_45_adam_private__transformer_block_private__multi_head_self_attention_dense_3_kernel_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:Ö
AssignVariableOp_46AssignVariableOpeassignvariableop_46_adam_private__transformer_block_private__multi_head_self_attention_dense_3_bias_mIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_47AssignVariableOp)assignvariableop_47_adam_dense_4_kernel_mIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_48AssignVariableOp'assignvariableop_48_adam_dense_4_bias_mIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_49AssignVariableOp)assignvariableop_49_adam_dense_5_kernel_mIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_50AssignVariableOp'assignvariableop_50_adam_dense_5_bias_mIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:À
AssignVariableOp_51AssignVariableOpOassignvariableop_51_adam_private__transformer_block_layer_normalization_gamma_mIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:¿
AssignVariableOp_52AssignVariableOpNassignvariableop_52_adam_private__transformer_block_layer_normalization_beta_mIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:Â
AssignVariableOp_53AssignVariableOpQassignvariableop_53_adam_private__transformer_block_layer_normalization_1_gamma_mIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:Á
AssignVariableOp_54AssignVariableOpPassignvariableop_54_adam_private__transformer_block_layer_normalization_1_beta_mIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_55AssignVariableOp)assignvariableop_55_adam_dense_6_kernel_vIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_56AssignVariableOp'assignvariableop_56_adam_dense_6_bias_vIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_57AssignVariableOp)assignvariableop_57_adam_dense_7_kernel_vIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_58AssignVariableOp'assignvariableop_58_adam_dense_7_bias_vIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:Æ
AssignVariableOp_59AssignVariableOpUassignvariableop_59_adam_private__token_and_position_embedding_embedding_embeddings_vIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:È
AssignVariableOp_60AssignVariableOpWassignvariableop_60_adam_private__token_and_position_embedding_embedding_1_embeddings_vIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:Ö
AssignVariableOp_61AssignVariableOpeassignvariableop_61_adam_private__transformer_block_private__multi_head_self_attention_dense_kernel_vIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:Ô
AssignVariableOp_62AssignVariableOpcassignvariableop_62_adam_private__transformer_block_private__multi_head_self_attention_dense_bias_vIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:Ø
AssignVariableOp_63AssignVariableOpgassignvariableop_63_adam_private__transformer_block_private__multi_head_self_attention_dense_1_kernel_vIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:Ö
AssignVariableOp_64AssignVariableOpeassignvariableop_64_adam_private__transformer_block_private__multi_head_self_attention_dense_1_bias_vIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:Ø
AssignVariableOp_65AssignVariableOpgassignvariableop_65_adam_private__transformer_block_private__multi_head_self_attention_dense_2_kernel_vIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:Ö
AssignVariableOp_66AssignVariableOpeassignvariableop_66_adam_private__transformer_block_private__multi_head_self_attention_dense_2_bias_vIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:Ø
AssignVariableOp_67AssignVariableOpgassignvariableop_67_adam_private__transformer_block_private__multi_head_self_attention_dense_3_kernel_vIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:Ö
AssignVariableOp_68AssignVariableOpeassignvariableop_68_adam_private__transformer_block_private__multi_head_self_attention_dense_3_bias_vIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_69AssignVariableOp)assignvariableop_69_adam_dense_4_kernel_vIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_70AssignVariableOp'assignvariableop_70_adam_dense_4_bias_vIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_71AssignVariableOp)assignvariableop_71_adam_dense_5_kernel_vIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_72AssignVariableOp'assignvariableop_72_adam_dense_5_bias_vIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:À
AssignVariableOp_73AssignVariableOpOassignvariableop_73_adam_private__transformer_block_layer_normalization_gamma_vIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:¿
AssignVariableOp_74AssignVariableOpNassignvariableop_74_adam_private__transformer_block_layer_normalization_beta_vIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:Â
AssignVariableOp_75AssignVariableOpQassignvariableop_75_adam_private__transformer_block_layer_normalization_1_gamma_vIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:Á
AssignVariableOp_76AssignVariableOpPassignvariableop_76_adam_private__transformer_block_layer_normalization_1_beta_vIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 í
Identity_77Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_78IdentityIdentity_77:output:0^NoOp_1*
T0*
_output_shapes
: Ú
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_78Identity_78:output:0*±
_input_shapes
: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_76AssignVariableOp_762(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix


ò
A__inference_dense_7_layer_call_and_return_conditional_losses_7482

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ä

9__inference_private__transformer_block_layer_call_fn_6838

inputs
unknown:  
	unknown_0: 
	unknown_1:  
	unknown_2: 
	unknown_3:  
	unknown_4: 
	unknown_5:  
	unknown_6: 
	unknown_7: 
	unknown_8: 
	unknown_9:  

unknown_10: 

unknown_11:  

unknown_12: 

unknown_13: 

unknown_14: 
identity¢StatefulPartitionedCall©
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_private__transformer_block_layer_call_and_return_conditional_losses_5169t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:ÿÿÿÿÿÿÿÿÿ : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
¼

&__inference_dense_6_layer_call_fn_7424

inputs
unknown: 
	unknown_0:
identity¢StatefulPartitionedCallÖ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dense_6_layer_call_and_return_conditional_losses_5222o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
ª
Ì
)__inference_sequential_layer_call_fn_7495

inputs
unknown:  
	unknown_0: 
	unknown_1:  
	unknown_2: 
identity¢StatefulPartitionedCallø
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_4763t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs

n
R__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_7388

inputs
identityX
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :p
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ^
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
±­
0
__inference__traced_save_7955
file_prefix-
)savev2_dense_6_kernel_read_readvariableop+
'savev2_dense_6_bias_read_readvariableop-
)savev2_dense_7_kernel_read_readvariableop+
'savev2_dense_7_bias_read_readvariableopY
Usavev2_private__token_and_position_embedding_embedding_embeddings_read_readvariableop[
Wsavev2_private__token_and_position_embedding_embedding_1_embeddings_read_readvariableopi
esavev2_private__transformer_block_private__multi_head_self_attention_dense_kernel_read_readvariableopg
csavev2_private__transformer_block_private__multi_head_self_attention_dense_bias_read_readvariableopk
gsavev2_private__transformer_block_private__multi_head_self_attention_dense_1_kernel_read_readvariableopi
esavev2_private__transformer_block_private__multi_head_self_attention_dense_1_bias_read_readvariableopk
gsavev2_private__transformer_block_private__multi_head_self_attention_dense_2_kernel_read_readvariableopi
esavev2_private__transformer_block_private__multi_head_self_attention_dense_2_bias_read_readvariableopk
gsavev2_private__transformer_block_private__multi_head_self_attention_dense_3_kernel_read_readvariableopi
esavev2_private__transformer_block_private__multi_head_self_attention_dense_3_bias_read_readvariableop-
)savev2_dense_4_kernel_read_readvariableop+
'savev2_dense_4_bias_read_readvariableop-
)savev2_dense_5_kernel_read_readvariableop+
'savev2_dense_5_bias_read_readvariableopS
Osavev2_private__transformer_block_layer_normalization_gamma_read_readvariableopR
Nsavev2_private__transformer_block_layer_normalization_beta_read_readvariableopU
Qsavev2_private__transformer_block_layer_normalization_1_gamma_read_readvariableopT
Psavev2_private__transformer_block_layer_normalization_1_beta_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop-
)savev2_true_positives_read_readvariableop-
)savev2_true_negatives_read_readvariableop.
*savev2_false_positives_read_readvariableop.
*savev2_false_negatives_read_readvariableop4
0savev2_adam_dense_6_kernel_m_read_readvariableop2
.savev2_adam_dense_6_bias_m_read_readvariableop4
0savev2_adam_dense_7_kernel_m_read_readvariableop2
.savev2_adam_dense_7_bias_m_read_readvariableop`
\savev2_adam_private__token_and_position_embedding_embedding_embeddings_m_read_readvariableopb
^savev2_adam_private__token_and_position_embedding_embedding_1_embeddings_m_read_readvariableopp
lsavev2_adam_private__transformer_block_private__multi_head_self_attention_dense_kernel_m_read_readvariableopn
jsavev2_adam_private__transformer_block_private__multi_head_self_attention_dense_bias_m_read_readvariableopr
nsavev2_adam_private__transformer_block_private__multi_head_self_attention_dense_1_kernel_m_read_readvariableopp
lsavev2_adam_private__transformer_block_private__multi_head_self_attention_dense_1_bias_m_read_readvariableopr
nsavev2_adam_private__transformer_block_private__multi_head_self_attention_dense_2_kernel_m_read_readvariableopp
lsavev2_adam_private__transformer_block_private__multi_head_self_attention_dense_2_bias_m_read_readvariableopr
nsavev2_adam_private__transformer_block_private__multi_head_self_attention_dense_3_kernel_m_read_readvariableopp
lsavev2_adam_private__transformer_block_private__multi_head_self_attention_dense_3_bias_m_read_readvariableop4
0savev2_adam_dense_4_kernel_m_read_readvariableop2
.savev2_adam_dense_4_bias_m_read_readvariableop4
0savev2_adam_dense_5_kernel_m_read_readvariableop2
.savev2_adam_dense_5_bias_m_read_readvariableopZ
Vsavev2_adam_private__transformer_block_layer_normalization_gamma_m_read_readvariableopY
Usavev2_adam_private__transformer_block_layer_normalization_beta_m_read_readvariableop\
Xsavev2_adam_private__transformer_block_layer_normalization_1_gamma_m_read_readvariableop[
Wsavev2_adam_private__transformer_block_layer_normalization_1_beta_m_read_readvariableop4
0savev2_adam_dense_6_kernel_v_read_readvariableop2
.savev2_adam_dense_6_bias_v_read_readvariableop4
0savev2_adam_dense_7_kernel_v_read_readvariableop2
.savev2_adam_dense_7_bias_v_read_readvariableop`
\savev2_adam_private__token_and_position_embedding_embedding_embeddings_v_read_readvariableopb
^savev2_adam_private__token_and_position_embedding_embedding_1_embeddings_v_read_readvariableopp
lsavev2_adam_private__transformer_block_private__multi_head_self_attention_dense_kernel_v_read_readvariableopn
jsavev2_adam_private__transformer_block_private__multi_head_self_attention_dense_bias_v_read_readvariableopr
nsavev2_adam_private__transformer_block_private__multi_head_self_attention_dense_1_kernel_v_read_readvariableopp
lsavev2_adam_private__transformer_block_private__multi_head_self_attention_dense_1_bias_v_read_readvariableopr
nsavev2_adam_private__transformer_block_private__multi_head_self_attention_dense_2_kernel_v_read_readvariableopp
lsavev2_adam_private__transformer_block_private__multi_head_self_attention_dense_2_bias_v_read_readvariableopr
nsavev2_adam_private__transformer_block_private__multi_head_self_attention_dense_3_kernel_v_read_readvariableopp
lsavev2_adam_private__transformer_block_private__multi_head_self_attention_dense_3_bias_v_read_readvariableop4
0savev2_adam_dense_4_kernel_v_read_readvariableop2
.savev2_adam_dense_4_bias_v_read_readvariableop4
0savev2_adam_dense_5_kernel_v_read_readvariableop2
.savev2_adam_dense_5_bias_v_read_readvariableopZ
Vsavev2_adam_private__transformer_block_layer_normalization_gamma_v_read_readvariableopY
Usavev2_adam_private__transformer_block_layer_normalization_beta_v_read_readvariableop\
Xsavev2_adam_private__transformer_block_layer_normalization_1_gamma_v_read_readvariableop[
Wsavev2_adam_private__transformer_block_layer_normalization_1_beta_v_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpointsw
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
: Å%
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:N*
dtype0*î$
valueä$Bá$NB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/1/true_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/1/true_negatives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/1/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/1/false_negatives/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:N*
dtype0*±
value§B¤NB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B õ.
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0)savev2_dense_6_kernel_read_readvariableop'savev2_dense_6_bias_read_readvariableop)savev2_dense_7_kernel_read_readvariableop'savev2_dense_7_bias_read_readvariableopUsavev2_private__token_and_position_embedding_embedding_embeddings_read_readvariableopWsavev2_private__token_and_position_embedding_embedding_1_embeddings_read_readvariableopesavev2_private__transformer_block_private__multi_head_self_attention_dense_kernel_read_readvariableopcsavev2_private__transformer_block_private__multi_head_self_attention_dense_bias_read_readvariableopgsavev2_private__transformer_block_private__multi_head_self_attention_dense_1_kernel_read_readvariableopesavev2_private__transformer_block_private__multi_head_self_attention_dense_1_bias_read_readvariableopgsavev2_private__transformer_block_private__multi_head_self_attention_dense_2_kernel_read_readvariableopesavev2_private__transformer_block_private__multi_head_self_attention_dense_2_bias_read_readvariableopgsavev2_private__transformer_block_private__multi_head_self_attention_dense_3_kernel_read_readvariableopesavev2_private__transformer_block_private__multi_head_self_attention_dense_3_bias_read_readvariableop)savev2_dense_4_kernel_read_readvariableop'savev2_dense_4_bias_read_readvariableop)savev2_dense_5_kernel_read_readvariableop'savev2_dense_5_bias_read_readvariableopOsavev2_private__transformer_block_layer_normalization_gamma_read_readvariableopNsavev2_private__transformer_block_layer_normalization_beta_read_readvariableopQsavev2_private__transformer_block_layer_normalization_1_gamma_read_readvariableopPsavev2_private__transformer_block_layer_normalization_1_beta_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop)savev2_true_positives_read_readvariableop)savev2_true_negatives_read_readvariableop*savev2_false_positives_read_readvariableop*savev2_false_negatives_read_readvariableop0savev2_adam_dense_6_kernel_m_read_readvariableop.savev2_adam_dense_6_bias_m_read_readvariableop0savev2_adam_dense_7_kernel_m_read_readvariableop.savev2_adam_dense_7_bias_m_read_readvariableop\savev2_adam_private__token_and_position_embedding_embedding_embeddings_m_read_readvariableop^savev2_adam_private__token_and_position_embedding_embedding_1_embeddings_m_read_readvariableoplsavev2_adam_private__transformer_block_private__multi_head_self_attention_dense_kernel_m_read_readvariableopjsavev2_adam_private__transformer_block_private__multi_head_self_attention_dense_bias_m_read_readvariableopnsavev2_adam_private__transformer_block_private__multi_head_self_attention_dense_1_kernel_m_read_readvariableoplsavev2_adam_private__transformer_block_private__multi_head_self_attention_dense_1_bias_m_read_readvariableopnsavev2_adam_private__transformer_block_private__multi_head_self_attention_dense_2_kernel_m_read_readvariableoplsavev2_adam_private__transformer_block_private__multi_head_self_attention_dense_2_bias_m_read_readvariableopnsavev2_adam_private__transformer_block_private__multi_head_self_attention_dense_3_kernel_m_read_readvariableoplsavev2_adam_private__transformer_block_private__multi_head_self_attention_dense_3_bias_m_read_readvariableop0savev2_adam_dense_4_kernel_m_read_readvariableop.savev2_adam_dense_4_bias_m_read_readvariableop0savev2_adam_dense_5_kernel_m_read_readvariableop.savev2_adam_dense_5_bias_m_read_readvariableopVsavev2_adam_private__transformer_block_layer_normalization_gamma_m_read_readvariableopUsavev2_adam_private__transformer_block_layer_normalization_beta_m_read_readvariableopXsavev2_adam_private__transformer_block_layer_normalization_1_gamma_m_read_readvariableopWsavev2_adam_private__transformer_block_layer_normalization_1_beta_m_read_readvariableop0savev2_adam_dense_6_kernel_v_read_readvariableop.savev2_adam_dense_6_bias_v_read_readvariableop0savev2_adam_dense_7_kernel_v_read_readvariableop.savev2_adam_dense_7_bias_v_read_readvariableop\savev2_adam_private__token_and_position_embedding_embedding_embeddings_v_read_readvariableop^savev2_adam_private__token_and_position_embedding_embedding_1_embeddings_v_read_readvariableoplsavev2_adam_private__transformer_block_private__multi_head_self_attention_dense_kernel_v_read_readvariableopjsavev2_adam_private__transformer_block_private__multi_head_self_attention_dense_bias_v_read_readvariableopnsavev2_adam_private__transformer_block_private__multi_head_self_attention_dense_1_kernel_v_read_readvariableoplsavev2_adam_private__transformer_block_private__multi_head_self_attention_dense_1_bias_v_read_readvariableopnsavev2_adam_private__transformer_block_private__multi_head_self_attention_dense_2_kernel_v_read_readvariableoplsavev2_adam_private__transformer_block_private__multi_head_self_attention_dense_2_bias_v_read_readvariableopnsavev2_adam_private__transformer_block_private__multi_head_self_attention_dense_3_kernel_v_read_readvariableoplsavev2_adam_private__transformer_block_private__multi_head_self_attention_dense_3_bias_v_read_readvariableop0savev2_adam_dense_4_kernel_v_read_readvariableop.savev2_adam_dense_4_bias_v_read_readvariableop0savev2_adam_dense_5_kernel_v_read_readvariableop.savev2_adam_dense_5_bias_v_read_readvariableopVsavev2_adam_private__transformer_block_layer_normalization_gamma_v_read_readvariableopUsavev2_adam_private__transformer_block_layer_normalization_beta_v_read_readvariableopXsavev2_adam_private__transformer_block_layer_normalization_1_gamma_v_read_readvariableopWsavev2_adam_private__transformer_block_layer_normalization_1_beta_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *\
dtypesR
P2N	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
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

identity_1Identity_1:output:0*Í
_input_shapes»
¸: : ::::	Æ :	 :  : :  : :  : :  : :  : :  : : : : : : : : : : : : :È:È:È:È: ::::	Æ :	 :  : :  : :  : :  : :  : :  : : : : : : ::::	Æ :	 :  : :  : :  : :  : :  : :  : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

: : 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::%!

_output_shapes
:	Æ :%!

_output_shapes
:	 :$ 

_output_shapes

:  : 

_output_shapes
: :$	 

_output_shapes

:  : 


_output_shapes
: :$ 

_output_shapes

:  : 

_output_shapes
: :$ 

_output_shapes

:  : 

_output_shapes
: :$ 

_output_shapes

:  : 

_output_shapes
: :$ 

_output_shapes

:  : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :!

_output_shapes	
:È:!

_output_shapes	
:È:! 

_output_shapes	
:È:!!

_output_shapes	
:È:$" 

_output_shapes

: : #

_output_shapes
::$$ 

_output_shapes

:: %

_output_shapes
::%&!

_output_shapes
:	Æ :%'!

_output_shapes
:	 :$( 

_output_shapes

:  : )

_output_shapes
: :$* 

_output_shapes

:  : +

_output_shapes
: :$, 

_output_shapes

:  : -

_output_shapes
: :$. 

_output_shapes

:  : /

_output_shapes
: :$0 

_output_shapes

:  : 1

_output_shapes
: :$2 

_output_shapes

:  : 3

_output_shapes
: : 4

_output_shapes
: : 5

_output_shapes
: : 6

_output_shapes
: : 7

_output_shapes
: :$8 

_output_shapes

: : 9

_output_shapes
::$: 

_output_shapes

:: ;

_output_shapes
::%<!

_output_shapes
:	Æ :%=!

_output_shapes
:	 :$> 

_output_shapes

:  : ?

_output_shapes
: :$@ 

_output_shapes

:  : A

_output_shapes
: :$B 

_output_shapes

:  : C

_output_shapes
: :$D 

_output_shapes

:  : E

_output_shapes
: :$F 

_output_shapes

:  : G

_output_shapes
: :$H 

_output_shapes

:  : I

_output_shapes
: : J

_output_shapes
: : K

_output_shapes
: : L

_output_shapes
: : M

_output_shapes
: :N

_output_shapes
: 
Î
ø
A__inference_dense_5_layer_call_and_return_conditional_losses_4756

inputs3
!tensordot_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢Tensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:  *
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
value	B : »
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
value	B : ¿
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
:ÿÿÿÿÿÿÿÿÿ 
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ [
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: Y
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
:ÿÿÿÿÿÿÿÿÿ r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ d
IdentityIdentityBiasAdd:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
ê

___inference_private__token_and_position_embedding_layer_call_and_return_conditional_losses_6801
x4
!embedding_1_embedding_lookup_6788:	 2
embedding_embedding_lookup_6794:	Æ 
identity¢embedding/embedding_lookup¢embedding_1/embedding_lookup6
ShapeShapex*
T0*
_output_shapes
:f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskM
range/startConst*
_output_shapes
: *
dtype0*
value	B : M
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :o
rangeRangerange/start:output:0strided_slice:output:0range/delta:output:0*
_output_shapes	
:Õ
embedding_1/embedding_lookupResourceGather!embedding_1_embedding_lookup_6788range:output:0*
Tindices0*4
_class*
(&loc:@embedding_1/embedding_lookup/6788*
_output_shapes
:	 *
dtype0¸
%embedding_1/embedding_lookup/IdentityIdentity%embedding_1/embedding_lookup:output:0*
T0*4
_class*
(&loc:@embedding_1/embedding_lookup/6788*
_output_shapes
:	 
'embedding_1/embedding_lookup/Identity_1Identity.embedding_1/embedding_lookup/Identity:output:0*
T0*
_output_shapes
:	 [
embedding/CastCastx*

DstT0*

SrcT0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà
embedding/embedding_lookupResourceGatherembedding_embedding_lookup_6794embedding/Cast:y:0*
Tindices0*2
_class(
&$loc:@embedding/embedding_lookup/6794*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
dtype0¿
#embedding/embedding_lookup/IdentityIdentity#embedding/embedding_lookup:output:0*
T0*2
_class(
&$loc:@embedding/embedding_lookup/6794*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
%embedding/embedding_lookup/Identity_1Identity,embedding/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¥
addAddV2.embedding/embedding_lookup/Identity_1:output:00embedding_1/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ [
IdentityIdentityadd:z:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
NoOpNoOp^embedding/embedding_lookup^embedding_1/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 28
embedding/embedding_lookupembedding/embedding_lookup2<
embedding_1/embedding_lookupembedding_1/embedding_lookup:K G
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_namex
Ø>
Ó
D__inference_sequential_layer_call_and_return_conditional_losses_7622

inputs;
)dense_4_tensordot_readvariableop_resource:  5
'dense_4_biasadd_readvariableop_resource: ;
)dense_5_tensordot_readvariableop_resource:  5
'dense_5_biasadd_readvariableop_resource: 
identity¢dense_4/BiasAdd/ReadVariableOp¢ dense_4/Tensordot/ReadVariableOp¢dense_5/BiasAdd/ReadVariableOp¢ dense_5/Tensordot/ReadVariableOp
 dense_4/Tensordot/ReadVariableOpReadVariableOp)dense_4_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype0`
dense_4/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:g
dense_4/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       M
dense_4/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:a
dense_4/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : Û
dense_4/Tensordot/GatherV2GatherV2 dense_4/Tensordot/Shape:output:0dense_4/Tensordot/free:output:0(dense_4/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:c
!dense_4/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ß
dense_4/Tensordot/GatherV2_1GatherV2 dense_4/Tensordot/Shape:output:0dense_4/Tensordot/axes:output:0*dense_4/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense_4/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
dense_4/Tensordot/ProdProd#dense_4/Tensordot/GatherV2:output:0 dense_4/Tensordot/Const:output:0*
T0*
_output_shapes
: c
dense_4/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
dense_4/Tensordot/Prod_1Prod%dense_4/Tensordot/GatherV2_1:output:0"dense_4/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
dense_4/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ¼
dense_4/Tensordot/concatConcatV2dense_4/Tensordot/free:output:0dense_4/Tensordot/axes:output:0&dense_4/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:
dense_4/Tensordot/stackPackdense_4/Tensordot/Prod:output:0!dense_4/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:
dense_4/Tensordot/transpose	Transposeinputs!dense_4/Tensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¢
dense_4/Tensordot/ReshapeReshapedense_4/Tensordot/transpose:y:0 dense_4/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¢
dense_4/Tensordot/MatMulMatMul"dense_4/Tensordot/Reshape:output:0(dense_4/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ c
dense_4/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: a
dense_4/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ç
dense_4/Tensordot/concat_1ConcatV2#dense_4/Tensordot/GatherV2:output:0"dense_4/Tensordot/Const_2:output:0(dense_4/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
dense_4/TensordotReshape"dense_4/Tensordot/MatMul:product:0#dense_4/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
dense_4/BiasAddBiasAdddense_4/Tensordot:output:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ e
dense_4/ReluReludense_4/BiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 dense_5/Tensordot/ReadVariableOpReadVariableOp)dense_5_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype0`
dense_5/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:g
dense_5/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       a
dense_5/Tensordot/ShapeShapedense_4/Relu:activations:0*
T0*
_output_shapes
:a
dense_5/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : Û
dense_5/Tensordot/GatherV2GatherV2 dense_5/Tensordot/Shape:output:0dense_5/Tensordot/free:output:0(dense_5/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:c
!dense_5/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ß
dense_5/Tensordot/GatherV2_1GatherV2 dense_5/Tensordot/Shape:output:0dense_5/Tensordot/axes:output:0*dense_5/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense_5/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
dense_5/Tensordot/ProdProd#dense_5/Tensordot/GatherV2:output:0 dense_5/Tensordot/Const:output:0*
T0*
_output_shapes
: c
dense_5/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
dense_5/Tensordot/Prod_1Prod%dense_5/Tensordot/GatherV2_1:output:0"dense_5/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
dense_5/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ¼
dense_5/Tensordot/concatConcatV2dense_5/Tensordot/free:output:0dense_5/Tensordot/axes:output:0&dense_5/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:
dense_5/Tensordot/stackPackdense_5/Tensordot/Prod:output:0!dense_5/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:
dense_5/Tensordot/transpose	Transposedense_4/Relu:activations:0!dense_5/Tensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¢
dense_5/Tensordot/ReshapeReshapedense_5/Tensordot/transpose:y:0 dense_5/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¢
dense_5/Tensordot/MatMulMatMul"dense_5/Tensordot/Reshape:output:0(dense_5/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ c
dense_5/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: a
dense_5/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ç
dense_5/Tensordot/concat_1ConcatV2#dense_5/Tensordot/GatherV2:output:0"dense_5/Tensordot/Const_2:output:0(dense_5/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
dense_5/TensordotReshape"dense_5/Tensordot/MatMul:product:0#dense_5/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
dense_5/BiasAddBiasAdddense_5/Tensordot:output:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ l
IdentityIdentitydense_5/BiasAdd:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Î
NoOpNoOp^dense_4/BiasAdd/ReadVariableOp!^dense_4/Tensordot/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp!^dense_5/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ : : : : 2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2D
 dense_4/Tensordot/ReadVariableOp dense_4/Tensordot/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2D
 dense_5/Tensordot/ReadVariableOp dense_5/Tensordot/ReadVariableOp:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Ñ
«
T__inference_private__transformer_block_layer_call_and_return_conditional_losses_7377

inputs\
Jprivate__multi_head_self_attention_dense_tensordot_readvariableop_resource:  V
Hprivate__multi_head_self_attention_dense_biasadd_readvariableop_resource: ^
Lprivate__multi_head_self_attention_dense_1_tensordot_readvariableop_resource:  X
Jprivate__multi_head_self_attention_dense_1_biasadd_readvariableop_resource: ^
Lprivate__multi_head_self_attention_dense_2_tensordot_readvariableop_resource:  X
Jprivate__multi_head_self_attention_dense_2_biasadd_readvariableop_resource: ^
Lprivate__multi_head_self_attention_dense_3_tensordot_readvariableop_resource:  X
Jprivate__multi_head_self_attention_dense_3_biasadd_readvariableop_resource: G
9layer_normalization_batchnorm_mul_readvariableop_resource: C
5layer_normalization_batchnorm_readvariableop_resource: F
4sequential_dense_4_tensordot_readvariableop_resource:  @
2sequential_dense_4_biasadd_readvariableop_resource: F
4sequential_dense_5_tensordot_readvariableop_resource:  @
2sequential_dense_5_biasadd_readvariableop_resource: I
;layer_normalization_1_batchnorm_mul_readvariableop_resource: E
7layer_normalization_1_batchnorm_readvariableop_resource: 
identity¢,layer_normalization/batchnorm/ReadVariableOp¢0layer_normalization/batchnorm/mul/ReadVariableOp¢.layer_normalization_1/batchnorm/ReadVariableOp¢2layer_normalization_1/batchnorm/mul/ReadVariableOp¢?private__multi_head_self_attention/dense/BiasAdd/ReadVariableOp¢Aprivate__multi_head_self_attention/dense/Tensordot/ReadVariableOp¢Aprivate__multi_head_self_attention/dense_1/BiasAdd/ReadVariableOp¢Cprivate__multi_head_self_attention/dense_1/Tensordot/ReadVariableOp¢Aprivate__multi_head_self_attention/dense_2/BiasAdd/ReadVariableOp¢Cprivate__multi_head_self_attention/dense_2/Tensordot/ReadVariableOp¢Aprivate__multi_head_self_attention/dense_3/BiasAdd/ReadVariableOp¢Cprivate__multi_head_self_attention/dense_3/Tensordot/ReadVariableOp¢)sequential/dense_4/BiasAdd/ReadVariableOp¢+sequential/dense_4/Tensordot/ReadVariableOp¢)sequential/dense_5/BiasAdd/ReadVariableOp¢+sequential/dense_5/Tensordot/ReadVariableOp^
(private__multi_head_self_attention/ShapeShapeinputs*
T0*
_output_shapes
:
6private__multi_head_self_attention/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
8private__multi_head_self_attention/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
8private__multi_head_self_attention/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
0private__multi_head_self_attention/strided_sliceStridedSlice1private__multi_head_self_attention/Shape:output:0?private__multi_head_self_attention/strided_slice/stack:output:0Aprivate__multi_head_self_attention/strided_slice/stack_1:output:0Aprivate__multi_head_self_attention/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskÌ
Aprivate__multi_head_self_attention/dense/Tensordot/ReadVariableOpReadVariableOpJprivate__multi_head_self_attention_dense_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype0
7private__multi_head_self_attention/dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:
7private__multi_head_self_attention/dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       n
8private__multi_head_self_attention/dense/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:
@private__multi_head_self_attention/dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ß
;private__multi_head_self_attention/dense/Tensordot/GatherV2GatherV2Aprivate__multi_head_self_attention/dense/Tensordot/Shape:output:0@private__multi_head_self_attention/dense/Tensordot/free:output:0Iprivate__multi_head_self_attention/dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:
Bprivate__multi_head_self_attention/dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ã
=private__multi_head_self_attention/dense/Tensordot/GatherV2_1GatherV2Aprivate__multi_head_self_attention/dense/Tensordot/Shape:output:0@private__multi_head_self_attention/dense/Tensordot/axes:output:0Kprivate__multi_head_self_attention/dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:
8private__multi_head_self_attention/dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: é
7private__multi_head_self_attention/dense/Tensordot/ProdProdDprivate__multi_head_self_attention/dense/Tensordot/GatherV2:output:0Aprivate__multi_head_self_attention/dense/Tensordot/Const:output:0*
T0*
_output_shapes
: 
:private__multi_head_self_attention/dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ï
9private__multi_head_self_attention/dense/Tensordot/Prod_1ProdFprivate__multi_head_self_attention/dense/Tensordot/GatherV2_1:output:0Cprivate__multi_head_self_attention/dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 
>private__multi_head_self_attention/dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : À
9private__multi_head_self_attention/dense/Tensordot/concatConcatV2@private__multi_head_self_attention/dense/Tensordot/free:output:0@private__multi_head_self_attention/dense/Tensordot/axes:output:0Gprivate__multi_head_self_attention/dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:ô
8private__multi_head_self_attention/dense/Tensordot/stackPack@private__multi_head_self_attention/dense/Tensordot/Prod:output:0Bprivate__multi_head_self_attention/dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:Ì
<private__multi_head_self_attention/dense/Tensordot/transpose	TransposeinputsBprivate__multi_head_self_attention/dense/Tensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
:private__multi_head_self_attention/dense/Tensordot/ReshapeReshape@private__multi_head_self_attention/dense/Tensordot/transpose:y:0Aprivate__multi_head_self_attention/dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
9private__multi_head_self_attention/dense/Tensordot/MatMulMatMulCprivate__multi_head_self_attention/dense/Tensordot/Reshape:output:0Iprivate__multi_head_self_attention/dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
:private__multi_head_self_attention/dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 
@private__multi_head_self_attention/dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ë
;private__multi_head_self_attention/dense/Tensordot/concat_1ConcatV2Dprivate__multi_head_self_attention/dense/Tensordot/GatherV2:output:0Cprivate__multi_head_self_attention/dense/Tensordot/Const_2:output:0Iprivate__multi_head_self_attention/dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:ÿ
2private__multi_head_self_attention/dense/TensordotReshapeCprivate__multi_head_self_attention/dense/Tensordot/MatMul:product:0Dprivate__multi_head_self_attention/dense/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ä
?private__multi_head_self_attention/dense/BiasAdd/ReadVariableOpReadVariableOpHprivate__multi_head_self_attention_dense_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0ø
0private__multi_head_self_attention/dense/BiasAddBiasAdd;private__multi_head_self_attention/dense/Tensordot:output:0Gprivate__multi_head_self_attention/dense/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ð
Cprivate__multi_head_self_attention/dense_1/Tensordot/ReadVariableOpReadVariableOpLprivate__multi_head_self_attention_dense_1_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype0
9private__multi_head_self_attention/dense_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:
9private__multi_head_self_attention/dense_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       p
:private__multi_head_self_attention/dense_1/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:
Bprivate__multi_head_self_attention/dense_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ç
=private__multi_head_self_attention/dense_1/Tensordot/GatherV2GatherV2Cprivate__multi_head_self_attention/dense_1/Tensordot/Shape:output:0Bprivate__multi_head_self_attention/dense_1/Tensordot/free:output:0Kprivate__multi_head_self_attention/dense_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:
Dprivate__multi_head_self_attention/dense_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ë
?private__multi_head_self_attention/dense_1/Tensordot/GatherV2_1GatherV2Cprivate__multi_head_self_attention/dense_1/Tensordot/Shape:output:0Bprivate__multi_head_self_attention/dense_1/Tensordot/axes:output:0Mprivate__multi_head_self_attention/dense_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:
:private__multi_head_self_attention/dense_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ï
9private__multi_head_self_attention/dense_1/Tensordot/ProdProdFprivate__multi_head_self_attention/dense_1/Tensordot/GatherV2:output:0Cprivate__multi_head_self_attention/dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: 
<private__multi_head_self_attention/dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: õ
;private__multi_head_self_attention/dense_1/Tensordot/Prod_1ProdHprivate__multi_head_self_attention/dense_1/Tensordot/GatherV2_1:output:0Eprivate__multi_head_self_attention/dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 
@private__multi_head_self_attention/dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : È
;private__multi_head_self_attention/dense_1/Tensordot/concatConcatV2Bprivate__multi_head_self_attention/dense_1/Tensordot/free:output:0Bprivate__multi_head_self_attention/dense_1/Tensordot/axes:output:0Iprivate__multi_head_self_attention/dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:ú
:private__multi_head_self_attention/dense_1/Tensordot/stackPackBprivate__multi_head_self_attention/dense_1/Tensordot/Prod:output:0Dprivate__multi_head_self_attention/dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:Ð
>private__multi_head_self_attention/dense_1/Tensordot/transpose	TransposeinputsDprivate__multi_head_self_attention/dense_1/Tensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
<private__multi_head_self_attention/dense_1/Tensordot/ReshapeReshapeBprivate__multi_head_self_attention/dense_1/Tensordot/transpose:y:0Cprivate__multi_head_self_attention/dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
;private__multi_head_self_attention/dense_1/Tensordot/MatMulMatMulEprivate__multi_head_self_attention/dense_1/Tensordot/Reshape:output:0Kprivate__multi_head_self_attention/dense_1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
<private__multi_head_self_attention/dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 
Bprivate__multi_head_self_attention/dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ó
=private__multi_head_self_attention/dense_1/Tensordot/concat_1ConcatV2Fprivate__multi_head_self_attention/dense_1/Tensordot/GatherV2:output:0Eprivate__multi_head_self_attention/dense_1/Tensordot/Const_2:output:0Kprivate__multi_head_self_attention/dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
4private__multi_head_self_attention/dense_1/TensordotReshapeEprivate__multi_head_self_attention/dense_1/Tensordot/MatMul:product:0Fprivate__multi_head_self_attention/dense_1/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ È
Aprivate__multi_head_self_attention/dense_1/BiasAdd/ReadVariableOpReadVariableOpJprivate__multi_head_self_attention_dense_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0þ
2private__multi_head_self_attention/dense_1/BiasAddBiasAdd=private__multi_head_self_attention/dense_1/Tensordot:output:0Iprivate__multi_head_self_attention/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ð
Cprivate__multi_head_self_attention/dense_2/Tensordot/ReadVariableOpReadVariableOpLprivate__multi_head_self_attention_dense_2_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype0
9private__multi_head_self_attention/dense_2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:
9private__multi_head_self_attention/dense_2/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       p
:private__multi_head_self_attention/dense_2/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:
Bprivate__multi_head_self_attention/dense_2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ç
=private__multi_head_self_attention/dense_2/Tensordot/GatherV2GatherV2Cprivate__multi_head_self_attention/dense_2/Tensordot/Shape:output:0Bprivate__multi_head_self_attention/dense_2/Tensordot/free:output:0Kprivate__multi_head_self_attention/dense_2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:
Dprivate__multi_head_self_attention/dense_2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ë
?private__multi_head_self_attention/dense_2/Tensordot/GatherV2_1GatherV2Cprivate__multi_head_self_attention/dense_2/Tensordot/Shape:output:0Bprivate__multi_head_self_attention/dense_2/Tensordot/axes:output:0Mprivate__multi_head_self_attention/dense_2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:
:private__multi_head_self_attention/dense_2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ï
9private__multi_head_self_attention/dense_2/Tensordot/ProdProdFprivate__multi_head_self_attention/dense_2/Tensordot/GatherV2:output:0Cprivate__multi_head_self_attention/dense_2/Tensordot/Const:output:0*
T0*
_output_shapes
: 
<private__multi_head_self_attention/dense_2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: õ
;private__multi_head_self_attention/dense_2/Tensordot/Prod_1ProdHprivate__multi_head_self_attention/dense_2/Tensordot/GatherV2_1:output:0Eprivate__multi_head_self_attention/dense_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 
@private__multi_head_self_attention/dense_2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : È
;private__multi_head_self_attention/dense_2/Tensordot/concatConcatV2Bprivate__multi_head_self_attention/dense_2/Tensordot/free:output:0Bprivate__multi_head_self_attention/dense_2/Tensordot/axes:output:0Iprivate__multi_head_self_attention/dense_2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:ú
:private__multi_head_self_attention/dense_2/Tensordot/stackPackBprivate__multi_head_self_attention/dense_2/Tensordot/Prod:output:0Dprivate__multi_head_self_attention/dense_2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:Ð
>private__multi_head_self_attention/dense_2/Tensordot/transpose	TransposeinputsDprivate__multi_head_self_attention/dense_2/Tensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
<private__multi_head_self_attention/dense_2/Tensordot/ReshapeReshapeBprivate__multi_head_self_attention/dense_2/Tensordot/transpose:y:0Cprivate__multi_head_self_attention/dense_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
;private__multi_head_self_attention/dense_2/Tensordot/MatMulMatMulEprivate__multi_head_self_attention/dense_2/Tensordot/Reshape:output:0Kprivate__multi_head_self_attention/dense_2/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
<private__multi_head_self_attention/dense_2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 
Bprivate__multi_head_self_attention/dense_2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ó
=private__multi_head_self_attention/dense_2/Tensordot/concat_1ConcatV2Fprivate__multi_head_self_attention/dense_2/Tensordot/GatherV2:output:0Eprivate__multi_head_self_attention/dense_2/Tensordot/Const_2:output:0Kprivate__multi_head_self_attention/dense_2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
4private__multi_head_self_attention/dense_2/TensordotReshapeEprivate__multi_head_self_attention/dense_2/Tensordot/MatMul:product:0Fprivate__multi_head_self_attention/dense_2/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ È
Aprivate__multi_head_self_attention/dense_2/BiasAdd/ReadVariableOpReadVariableOpJprivate__multi_head_self_attention_dense_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0þ
2private__multi_head_self_attention/dense_2/BiasAddBiasAdd=private__multi_head_self_attention/dense_2/Tensordot:output:0Iprivate__multi_head_self_attention/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ }
2private__multi_head_self_attention/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿt
2private__multi_head_self_attention/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :t
2private__multi_head_self_attention/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :Ø
0private__multi_head_self_attention/Reshape/shapePack9private__multi_head_self_attention/strided_slice:output:0;private__multi_head_self_attention/Reshape/shape/1:output:0;private__multi_head_self_attention/Reshape/shape/2:output:0;private__multi_head_self_attention/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:î
*private__multi_head_self_attention/ReshapeReshape9private__multi_head_self_attention/dense/BiasAdd:output:09private__multi_head_self_attention/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
1private__multi_head_self_attention/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             í
,private__multi_head_self_attention/transpose	Transpose3private__multi_head_self_attention/Reshape:output:0:private__multi_head_self_attention/transpose/perm:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
4private__multi_head_self_attention/Reshape_1/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿv
4private__multi_head_self_attention/Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :v
4private__multi_head_self_attention/Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :à
2private__multi_head_self_attention/Reshape_1/shapePack9private__multi_head_self_attention/strided_slice:output:0=private__multi_head_self_attention/Reshape_1/shape/1:output:0=private__multi_head_self_attention/Reshape_1/shape/2:output:0=private__multi_head_self_attention/Reshape_1/shape/3:output:0*
N*
T0*
_output_shapes
:ô
,private__multi_head_self_attention/Reshape_1Reshape;private__multi_head_self_attention/dense_1/BiasAdd:output:0;private__multi_head_self_attention/Reshape_1/shape:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
3private__multi_head_self_attention/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             ó
.private__multi_head_self_attention/transpose_1	Transpose5private__multi_head_self_attention/Reshape_1:output:0<private__multi_head_self_attention/transpose_1/perm:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
4private__multi_head_self_attention/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿv
4private__multi_head_self_attention/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :v
4private__multi_head_self_attention/Reshape_2/shape/3Const*
_output_shapes
: *
dtype0*
value	B :à
2private__multi_head_self_attention/Reshape_2/shapePack9private__multi_head_self_attention/strided_slice:output:0=private__multi_head_self_attention/Reshape_2/shape/1:output:0=private__multi_head_self_attention/Reshape_2/shape/2:output:0=private__multi_head_self_attention/Reshape_2/shape/3:output:0*
N*
T0*
_output_shapes
:ô
,private__multi_head_self_attention/Reshape_2Reshape;private__multi_head_self_attention/dense_2/BiasAdd:output:0;private__multi_head_self_attention/Reshape_2/shape:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
3private__multi_head_self_attention/transpose_2/permConst*
_output_shapes
:*
dtype0*%
valueB"             ó
.private__multi_head_self_attention/transpose_2	Transpose5private__multi_head_self_attention/Reshape_2:output:0<private__multi_head_self_attention/transpose_2/perm:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿù
)private__multi_head_self_attention/MatMulBatchMatMulV20private__multi_head_self_attention/transpose:y:02private__multi_head_self_attention/transpose_1:y:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
adj_y(
*private__multi_head_self_attention/Shape_1Shape2private__multi_head_self_attention/transpose_1:y:0*
T0*
_output_shapes
:
8private__multi_head_self_attention/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ
:private__multi_head_self_attention/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
:private__multi_head_self_attention/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
2private__multi_head_self_attention/strided_slice_1StridedSlice3private__multi_head_self_attention/Shape_1:output:0Aprivate__multi_head_self_attention/strided_slice_1/stack:output:0Cprivate__multi_head_self_attention/strided_slice_1/stack_1:output:0Cprivate__multi_head_self_attention/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
'private__multi_head_self_attention/CastCast;private__multi_head_self_attention/strided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: }
'private__multi_head_self_attention/SqrtSqrt+private__multi_head_self_attention/Cast:y:0*
T0*
_output_shapes
: â
*private__multi_head_self_attention/truedivRealDiv2private__multi_head_self_attention/MatMul:output:0+private__multi_head_self_attention/Sqrt:y:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ±
*private__multi_head_self_attention/SoftmaxSoftmax.private__multi_head_self_attention/truediv:z:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿé
+private__multi_head_self_attention/MatMul_1BatchMatMulV24private__multi_head_self_attention/Softmax:softmax:02private__multi_head_self_attention/transpose_2:y:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
3private__multi_head_self_attention/transpose_3/permConst*
_output_shapes
:*
dtype0*%
valueB"             ò
.private__multi_head_self_attention/transpose_3	Transpose4private__multi_head_self_attention/MatMul_1:output:0<private__multi_head_self_attention/transpose_3/perm:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
4private__multi_head_self_attention/Reshape_3/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿv
4private__multi_head_self_attention/Reshape_3/shape/2Const*
_output_shapes
: *
dtype0*
value	B : ¡
2private__multi_head_self_attention/Reshape_3/shapePack9private__multi_head_self_attention/strided_slice:output:0=private__multi_head_self_attention/Reshape_3/shape/1:output:0=private__multi_head_self_attention/Reshape_3/shape/2:output:0*
N*
T0*
_output_shapes
:ç
,private__multi_head_self_attention/Reshape_3Reshape2private__multi_head_self_attention/transpose_3:y:0;private__multi_head_self_attention/Reshape_3/shape:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ Ð
Cprivate__multi_head_self_attention/dense_3/Tensordot/ReadVariableOpReadVariableOpLprivate__multi_head_self_attention_dense_3_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype0
9private__multi_head_self_attention/dense_3/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:
9private__multi_head_self_attention/dense_3/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       
:private__multi_head_self_attention/dense_3/Tensordot/ShapeShape5private__multi_head_self_attention/Reshape_3:output:0*
T0*
_output_shapes
:
Bprivate__multi_head_self_attention/dense_3/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ç
=private__multi_head_self_attention/dense_3/Tensordot/GatherV2GatherV2Cprivate__multi_head_self_attention/dense_3/Tensordot/Shape:output:0Bprivate__multi_head_self_attention/dense_3/Tensordot/free:output:0Kprivate__multi_head_self_attention/dense_3/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:
Dprivate__multi_head_self_attention/dense_3/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ë
?private__multi_head_self_attention/dense_3/Tensordot/GatherV2_1GatherV2Cprivate__multi_head_self_attention/dense_3/Tensordot/Shape:output:0Bprivate__multi_head_self_attention/dense_3/Tensordot/axes:output:0Mprivate__multi_head_self_attention/dense_3/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:
:private__multi_head_self_attention/dense_3/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ï
9private__multi_head_self_attention/dense_3/Tensordot/ProdProdFprivate__multi_head_self_attention/dense_3/Tensordot/GatherV2:output:0Cprivate__multi_head_self_attention/dense_3/Tensordot/Const:output:0*
T0*
_output_shapes
: 
<private__multi_head_self_attention/dense_3/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: õ
;private__multi_head_self_attention/dense_3/Tensordot/Prod_1ProdHprivate__multi_head_self_attention/dense_3/Tensordot/GatherV2_1:output:0Eprivate__multi_head_self_attention/dense_3/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 
@private__multi_head_self_attention/dense_3/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : È
;private__multi_head_self_attention/dense_3/Tensordot/concatConcatV2Bprivate__multi_head_self_attention/dense_3/Tensordot/free:output:0Bprivate__multi_head_self_attention/dense_3/Tensordot/axes:output:0Iprivate__multi_head_self_attention/dense_3/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:ú
:private__multi_head_self_attention/dense_3/Tensordot/stackPackBprivate__multi_head_self_attention/dense_3/Tensordot/Prod:output:0Dprivate__multi_head_self_attention/dense_3/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:
>private__multi_head_self_attention/dense_3/Tensordot/transpose	Transpose5private__multi_head_self_attention/Reshape_3:output:0Dprivate__multi_head_self_attention/dense_3/Tensordot/concat:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
<private__multi_head_self_attention/dense_3/Tensordot/ReshapeReshapeBprivate__multi_head_self_attention/dense_3/Tensordot/transpose:y:0Cprivate__multi_head_self_attention/dense_3/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
;private__multi_head_self_attention/dense_3/Tensordot/MatMulMatMulEprivate__multi_head_self_attention/dense_3/Tensordot/Reshape:output:0Kprivate__multi_head_self_attention/dense_3/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
<private__multi_head_self_attention/dense_3/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 
Bprivate__multi_head_self_attention/dense_3/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ó
=private__multi_head_self_attention/dense_3/Tensordot/concat_1ConcatV2Fprivate__multi_head_self_attention/dense_3/Tensordot/GatherV2:output:0Eprivate__multi_head_self_attention/dense_3/Tensordot/Const_2:output:0Kprivate__multi_head_self_attention/dense_3/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
4private__multi_head_self_attention/dense_3/TensordotReshapeEprivate__multi_head_self_attention/dense_3/Tensordot/MatMul:product:0Fprivate__multi_head_self_attention/dense_3/Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ È
Aprivate__multi_head_self_attention/dense_3/BiasAdd/ReadVariableOpReadVariableOpJprivate__multi_head_self_attention_dense_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
2private__multi_head_self_attention/dense_3/BiasAddBiasAdd=private__multi_head_self_attention/dense_3/Tensordot:output:0Iprivate__multi_head_self_attention/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ Z
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?¶
dropout/dropout/MulMul;private__multi_head_self_attention/dense_3/BiasAdd:output:0dropout/dropout/Const:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
dropout/dropout/ShapeShape;private__multi_head_self_attention/dense_3/BiasAdd:output:0*
T0*
_output_shapes
:©
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *
dtype0c
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=Ë
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ f
addAddV2inputsdropout/dropout/Mul_1:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ |
2layer_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:¶
 layer_normalization/moments/meanMeanadd:z:0;layer_normalization/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(
(layer_normalization/moments/StopGradientStopGradient)layer_normalization/moments/mean:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿµ
-layer_normalization/moments/SquaredDifferenceSquaredDifferenceadd:z:01layer_normalization/moments/StopGradient:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
6layer_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:è
$layer_normalization/moments/varianceMean1layer_normalization/moments/SquaredDifference:z:0?layer_normalization/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(h
#layer_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½75¾
!layer_normalization/batchnorm/addAddV2-layer_normalization/moments/variance:output:0,layer_normalization/batchnorm/add/y:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#layer_normalization/batchnorm/RsqrtRsqrt%layer_normalization/batchnorm/add:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¦
0layer_normalization/batchnorm/mul/ReadVariableOpReadVariableOp9layer_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype0Â
!layer_normalization/batchnorm/mulMul'layer_normalization/batchnorm/Rsqrt:y:08layer_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
#layer_normalization/batchnorm/mul_1Muladd:z:0%layer_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ³
#layer_normalization/batchnorm/mul_2Mul)layer_normalization/moments/mean:output:0%layer_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
,layer_normalization/batchnorm/ReadVariableOpReadVariableOp5layer_normalization_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype0¾
!layer_normalization/batchnorm/subSub4layer_normalization/batchnorm/ReadVariableOp:value:0'layer_normalization/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ³
#layer_normalization/batchnorm/add_1AddV2'layer_normalization/batchnorm/mul_1:z:0%layer_normalization/batchnorm/sub:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
+sequential/dense_4/Tensordot/ReadVariableOpReadVariableOp4sequential_dense_4_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype0k
!sequential/dense_4/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:r
!sequential/dense_4/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       y
"sequential/dense_4/Tensordot/ShapeShape'layer_normalization/batchnorm/add_1:z:0*
T0*
_output_shapes
:l
*sequential/dense_4/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
%sequential/dense_4/Tensordot/GatherV2GatherV2+sequential/dense_4/Tensordot/Shape:output:0*sequential/dense_4/Tensordot/free:output:03sequential/dense_4/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:n
,sequential/dense_4/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
'sequential/dense_4/Tensordot/GatherV2_1GatherV2+sequential/dense_4/Tensordot/Shape:output:0*sequential/dense_4/Tensordot/axes:output:05sequential/dense_4/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:l
"sequential/dense_4/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: §
!sequential/dense_4/Tensordot/ProdProd.sequential/dense_4/Tensordot/GatherV2:output:0+sequential/dense_4/Tensordot/Const:output:0*
T0*
_output_shapes
: n
$sequential/dense_4/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ­
#sequential/dense_4/Tensordot/Prod_1Prod0sequential/dense_4/Tensordot/GatherV2_1:output:0-sequential/dense_4/Tensordot/Const_1:output:0*
T0*
_output_shapes
: j
(sequential/dense_4/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : è
#sequential/dense_4/Tensordot/concatConcatV2*sequential/dense_4/Tensordot/free:output:0*sequential/dense_4/Tensordot/axes:output:01sequential/dense_4/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:²
"sequential/dense_4/Tensordot/stackPack*sequential/dense_4/Tensordot/Prod:output:0,sequential/dense_4/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:Á
&sequential/dense_4/Tensordot/transpose	Transpose'layer_normalization/batchnorm/add_1:z:0,sequential/dense_4/Tensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ã
$sequential/dense_4/Tensordot/ReshapeReshape*sequential/dense_4/Tensordot/transpose:y:0+sequential/dense_4/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÃ
#sequential/dense_4/Tensordot/MatMulMatMul-sequential/dense_4/Tensordot/Reshape:output:03sequential/dense_4/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ n
$sequential/dense_4/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: l
*sequential/dense_4/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ó
%sequential/dense_4/Tensordot/concat_1ConcatV2.sequential/dense_4/Tensordot/GatherV2:output:0-sequential/dense_4/Tensordot/Const_2:output:03sequential/dense_4/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:½
sequential/dense_4/TensordotReshape-sequential/dense_4/Tensordot/MatMul:product:0.sequential/dense_4/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
)sequential/dense_4/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0¶
sequential/dense_4/BiasAddBiasAdd%sequential/dense_4/Tensordot:output:01sequential/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ {
sequential/dense_4/ReluRelu#sequential/dense_4/BiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
+sequential/dense_5/Tensordot/ReadVariableOpReadVariableOp4sequential_dense_5_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype0k
!sequential/dense_5/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:r
!sequential/dense_5/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       w
"sequential/dense_5/Tensordot/ShapeShape%sequential/dense_4/Relu:activations:0*
T0*
_output_shapes
:l
*sequential/dense_5/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
%sequential/dense_5/Tensordot/GatherV2GatherV2+sequential/dense_5/Tensordot/Shape:output:0*sequential/dense_5/Tensordot/free:output:03sequential/dense_5/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:n
,sequential/dense_5/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
'sequential/dense_5/Tensordot/GatherV2_1GatherV2+sequential/dense_5/Tensordot/Shape:output:0*sequential/dense_5/Tensordot/axes:output:05sequential/dense_5/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:l
"sequential/dense_5/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: §
!sequential/dense_5/Tensordot/ProdProd.sequential/dense_5/Tensordot/GatherV2:output:0+sequential/dense_5/Tensordot/Const:output:0*
T0*
_output_shapes
: n
$sequential/dense_5/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ­
#sequential/dense_5/Tensordot/Prod_1Prod0sequential/dense_5/Tensordot/GatherV2_1:output:0-sequential/dense_5/Tensordot/Const_1:output:0*
T0*
_output_shapes
: j
(sequential/dense_5/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : è
#sequential/dense_5/Tensordot/concatConcatV2*sequential/dense_5/Tensordot/free:output:0*sequential/dense_5/Tensordot/axes:output:01sequential/dense_5/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:²
"sequential/dense_5/Tensordot/stackPack*sequential/dense_5/Tensordot/Prod:output:0,sequential/dense_5/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:¿
&sequential/dense_5/Tensordot/transpose	Transpose%sequential/dense_4/Relu:activations:0,sequential/dense_5/Tensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ã
$sequential/dense_5/Tensordot/ReshapeReshape*sequential/dense_5/Tensordot/transpose:y:0+sequential/dense_5/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÃ
#sequential/dense_5/Tensordot/MatMulMatMul-sequential/dense_5/Tensordot/Reshape:output:03sequential/dense_5/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ n
$sequential/dense_5/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: l
*sequential/dense_5/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ó
%sequential/dense_5/Tensordot/concat_1ConcatV2.sequential/dense_5/Tensordot/GatherV2:output:0-sequential/dense_5/Tensordot/Const_2:output:03sequential/dense_5/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:½
sequential/dense_5/TensordotReshape-sequential/dense_5/Tensordot/MatMul:product:0.sequential/dense_5/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
)sequential/dense_5/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0¶
sequential/dense_5/BiasAddBiasAdd%sequential/dense_5/Tensordot:output:01sequential/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ \
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?
dropout_1/dropout/MulMul#sequential/dense_5/BiasAdd:output:0 dropout_1/dropout/Const:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ j
dropout_1/dropout/ShapeShape#sequential/dense_5/BiasAdd:output:0*
T0*
_output_shapes
:¥
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
dtype0e
 dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=É
dropout_1/dropout/GreaterEqualGreaterEqual7dropout_1/dropout/random_uniform/RandomUniform:output:0)dropout_1/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
dropout_1/dropout/Mul_1Muldropout_1/dropout/Mul:z:0dropout_1/dropout/Cast:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
add_1AddV2'layer_normalization/batchnorm/add_1:z:0dropout_1/dropout/Mul_1:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ~
4layer_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:¼
"layer_normalization_1/moments/meanMean	add_1:z:0=layer_normalization_1/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(
*layer_normalization_1/moments/StopGradientStopGradient+layer_normalization_1/moments/mean:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ»
/layer_normalization_1/moments/SquaredDifferenceSquaredDifference	add_1:z:03layer_normalization_1/moments/StopGradient:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
8layer_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:î
&layer_normalization_1/moments/varianceMean3layer_normalization_1/moments/SquaredDifference:z:0Alayer_normalization_1/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(j
%layer_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½75Ä
#layer_normalization_1/batchnorm/addAddV2/layer_normalization_1/moments/variance:output:0.layer_normalization_1/batchnorm/add/y:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%layer_normalization_1/batchnorm/RsqrtRsqrt'layer_normalization_1/batchnorm/add:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
2layer_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype0È
#layer_normalization_1/batchnorm/mulMul)layer_normalization_1/batchnorm/Rsqrt:y:0:layer_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
%layer_normalization_1/batchnorm/mul_1Mul	add_1:z:0'layer_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¹
%layer_normalization_1/batchnorm/mul_2Mul+layer_normalization_1/moments/mean:output:0'layer_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¢
.layer_normalization_1/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype0Ä
#layer_normalization_1/batchnorm/subSub6layer_normalization_1/batchnorm/ReadVariableOp:value:0)layer_normalization_1/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¹
%layer_normalization_1/batchnorm/add_1AddV2)layer_normalization_1/batchnorm/mul_1:z:0'layer_normalization_1/batchnorm/sub:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ }
IdentityIdentity)layer_normalization_1/batchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ æ
NoOpNoOp-^layer_normalization/batchnorm/ReadVariableOp1^layer_normalization/batchnorm/mul/ReadVariableOp/^layer_normalization_1/batchnorm/ReadVariableOp3^layer_normalization_1/batchnorm/mul/ReadVariableOp@^private__multi_head_self_attention/dense/BiasAdd/ReadVariableOpB^private__multi_head_self_attention/dense/Tensordot/ReadVariableOpB^private__multi_head_self_attention/dense_1/BiasAdd/ReadVariableOpD^private__multi_head_self_attention/dense_1/Tensordot/ReadVariableOpB^private__multi_head_self_attention/dense_2/BiasAdd/ReadVariableOpD^private__multi_head_self_attention/dense_2/Tensordot/ReadVariableOpB^private__multi_head_self_attention/dense_3/BiasAdd/ReadVariableOpD^private__multi_head_self_attention/dense_3/Tensordot/ReadVariableOp*^sequential/dense_4/BiasAdd/ReadVariableOp,^sequential/dense_4/Tensordot/ReadVariableOp*^sequential/dense_5/BiasAdd/ReadVariableOp,^sequential/dense_5/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:ÿÿÿÿÿÿÿÿÿ : : : : : : : : : : : : : : : : 2\
,layer_normalization/batchnorm/ReadVariableOp,layer_normalization/batchnorm/ReadVariableOp2d
0layer_normalization/batchnorm/mul/ReadVariableOp0layer_normalization/batchnorm/mul/ReadVariableOp2`
.layer_normalization_1/batchnorm/ReadVariableOp.layer_normalization_1/batchnorm/ReadVariableOp2h
2layer_normalization_1/batchnorm/mul/ReadVariableOp2layer_normalization_1/batchnorm/mul/ReadVariableOp2
?private__multi_head_self_attention/dense/BiasAdd/ReadVariableOp?private__multi_head_self_attention/dense/BiasAdd/ReadVariableOp2
Aprivate__multi_head_self_attention/dense/Tensordot/ReadVariableOpAprivate__multi_head_self_attention/dense/Tensordot/ReadVariableOp2
Aprivate__multi_head_self_attention/dense_1/BiasAdd/ReadVariableOpAprivate__multi_head_self_attention/dense_1/BiasAdd/ReadVariableOp2
Cprivate__multi_head_self_attention/dense_1/Tensordot/ReadVariableOpCprivate__multi_head_self_attention/dense_1/Tensordot/ReadVariableOp2
Aprivate__multi_head_self_attention/dense_2/BiasAdd/ReadVariableOpAprivate__multi_head_self_attention/dense_2/BiasAdd/ReadVariableOp2
Cprivate__multi_head_self_attention/dense_2/Tensordot/ReadVariableOpCprivate__multi_head_self_attention/dense_2/Tensordot/ReadVariableOp2
Aprivate__multi_head_self_attention/dense_3/BiasAdd/ReadVariableOpAprivate__multi_head_self_attention/dense_3/BiasAdd/ReadVariableOp2
Cprivate__multi_head_self_attention/dense_3/Tensordot/ReadVariableOpCprivate__multi_head_self_attention/dense_3/Tensordot/ReadVariableOp2V
)sequential/dense_4/BiasAdd/ReadVariableOp)sequential/dense_4/BiasAdd/ReadVariableOp2Z
+sequential/dense_4/Tensordot/ReadVariableOp+sequential/dense_4/Tensordot/ReadVariableOp2V
)sequential/dense_5/BiasAdd/ReadVariableOp)sequential/dense_5/BiasAdd/ReadVariableOp2Z
+sequential/dense_5/Tensordot/ReadVariableOp+sequential/dense_5/Tensordot/ReadVariableOp:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
§
ø
A__inference_dense_4_layer_call_and_return_conditional_losses_4720

inputs3
!tensordot_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢Tensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:  *
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
value	B : »
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
value	B : ¿
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
:ÿÿÿÿÿÿÿÿÿ 
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ [
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: Y
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
:ÿÿÿÿÿÿÿÿÿ r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ U
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ f
IdentityIdentityRelu:activations:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
,


?__inference_model_layer_call_and_return_conditional_losses_5966
input_1=
*private__token_and_position_embedding_5914:	 =
*private__token_and_position_embedding_5916:	Æ 1
private__transformer_block_5919:  -
private__transformer_block_5921: 1
private__transformer_block_5923:  -
private__transformer_block_5925: 1
private__transformer_block_5927:  -
private__transformer_block_5929: 1
private__transformer_block_5931:  -
private__transformer_block_5933: -
private__transformer_block_5935: -
private__transformer_block_5937: 1
private__transformer_block_5939:  -
private__transformer_block_5941: 1
private__transformer_block_5943:  -
private__transformer_block_5945: -
private__transformer_block_5947: -
private__transformer_block_5949: 
dense_6_5954: 
dense_6_5956:
dense_7_5960:
dense_7_5962:
identity¢dense_6/StatefulPartitionedCall¢dense_7/StatefulPartitionedCall¢=private__token_and_position_embedding/StatefulPartitionedCall¢2private__transformer_block/StatefulPartitionedCallä
=private__token_and_position_embedding/StatefulPartitionedCallStatefulPartitionedCallinput_1*private__token_and_position_embedding_5914*private__token_and_position_embedding_5916*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *h
fcRa
___inference_private__token_and_position_embedding_layer_call_and_return_conditional_losses_4919á
2private__transformer_block/StatefulPartitionedCallStatefulPartitionedCallFprivate__token_and_position_embedding/StatefulPartitionedCall:output:0private__transformer_block_5919private__transformer_block_5921private__transformer_block_5923private__transformer_block_5925private__transformer_block_5927private__transformer_block_5929private__transformer_block_5931private__transformer_block_5933private__transformer_block_5935private__transformer_block_5937private__transformer_block_5939private__transformer_block_5941private__transformer_block_5943private__transformer_block_5945private__transformer_block_5947private__transformer_block_5949*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_private__transformer_block_layer_call_and_return_conditional_losses_5169
(global_average_pooling1d/PartitionedCallPartitionedCall;private__transformer_block/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_4885ã
dropout_2/PartitionedCallPartitionedCall1global_average_pooling1d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dropout_2_layer_call_and_return_conditional_losses_5209
dense_6/StatefulPartitionedCallStatefulPartitionedCall"dropout_2/PartitionedCall:output:0dense_6_5954dense_6_5956*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dense_6_layer_call_and_return_conditional_losses_5222Ú
dropout_3/PartitionedCallPartitionedCall(dense_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dropout_3_layer_call_and_return_conditional_losses_5233
dense_7/StatefulPartitionedCallStatefulPartitionedCall"dropout_3/PartitionedCall:output:0dense_7_5960dense_7_5962*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dense_7_layer_call_and_return_conditional_losses_5246w
IdentityIdentity(dense_7/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿ
NoOpNoOp ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall>^private__token_and_position_embedding/StatefulPartitionedCall3^private__transformer_block/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : 2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2~
=private__token_and_position_embedding/StatefulPartitionedCall=private__token_and_position_embedding/StatefulPartitionedCall2h
2private__transformer_block/StatefulPartitionedCall2private__transformer_block/StatefulPartitionedCall:Q M
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
/
Ý

?__inference_model_layer_call_and_return_conditional_losses_5815

inputs=
*private__token_and_position_embedding_5763:	 =
*private__token_and_position_embedding_5765:	Æ 1
private__transformer_block_5768:  -
private__transformer_block_5770: 1
private__transformer_block_5772:  -
private__transformer_block_5774: 1
private__transformer_block_5776:  -
private__transformer_block_5778: 1
private__transformer_block_5780:  -
private__transformer_block_5782: -
private__transformer_block_5784: -
private__transformer_block_5786: 1
private__transformer_block_5788:  -
private__transformer_block_5790: 1
private__transformer_block_5792:  -
private__transformer_block_5794: -
private__transformer_block_5796: -
private__transformer_block_5798: 
dense_6_5803: 
dense_6_5805:
dense_7_5809:
dense_7_5811:
identity¢dense_6/StatefulPartitionedCall¢dense_7/StatefulPartitionedCall¢!dropout_2/StatefulPartitionedCall¢!dropout_3/StatefulPartitionedCall¢=private__token_and_position_embedding/StatefulPartitionedCall¢2private__transformer_block/StatefulPartitionedCallã
=private__token_and_position_embedding/StatefulPartitionedCallStatefulPartitionedCallinputs*private__token_and_position_embedding_5763*private__token_and_position_embedding_5765*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *h
fcRa
___inference_private__token_and_position_embedding_layer_call_and_return_conditional_losses_4919á
2private__transformer_block/StatefulPartitionedCallStatefulPartitionedCallFprivate__token_and_position_embedding/StatefulPartitionedCall:output:0private__transformer_block_5768private__transformer_block_5770private__transformer_block_5772private__transformer_block_5774private__transformer_block_5776private__transformer_block_5778private__transformer_block_5780private__transformer_block_5782private__transformer_block_5784private__transformer_block_5786private__transformer_block_5788private__transformer_block_5790private__transformer_block_5792private__transformer_block_5794private__transformer_block_5796private__transformer_block_5798*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_private__transformer_block_layer_call_and_return_conditional_losses_5664
(global_average_pooling1d/PartitionedCallPartitionedCall;private__transformer_block/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_4885ó
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall1global_average_pooling1d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dropout_2_layer_call_and_return_conditional_losses_5363
dense_6/StatefulPartitionedCallStatefulPartitionedCall*dropout_2/StatefulPartitionedCall:output:0dense_6_5803dense_6_5805*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dense_6_layer_call_and_return_conditional_losses_5222
!dropout_3/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0"^dropout_2/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dropout_3_layer_call_and_return_conditional_losses_5330
dense_7/StatefulPartitionedCallStatefulPartitionedCall*dropout_3/StatefulPartitionedCall:output:0dense_7_5809dense_7_5811*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dense_7_layer_call_and_return_conditional_losses_5246w
IdentityIdentity(dense_7/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÇ
NoOpNoOp ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall"^dropout_3/StatefulPartitionedCall>^private__token_and_position_embedding/StatefulPartitionedCall3^private__transformer_block/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : 2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall2F
!dropout_3/StatefulPartitionedCall!dropout_3/StatefulPartitionedCall2~
=private__token_and_position_embedding/StatefulPartitionedCall=private__token_and_position_embedding/StatefulPartitionedCall2h
2private__transformer_block/StatefulPartitionedCall2private__transformer_block/StatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ö
a
C__inference_dropout_2_layer_call_and_return_conditional_losses_7403

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ [

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Ð

&__inference_dense_5_layer_call_fn_7671

inputs
unknown:  
	unknown_0: 
identity¢StatefulPartitionedCallÛ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dense_5_layer_call_and_return_conditional_losses_4756t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Ð

&__inference_dense_4_layer_call_fn_7631

inputs
unknown:  
	unknown_0: 
identity¢StatefulPartitionedCallÛ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dense_4_layer_call_and_return_conditional_losses_4720t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Ä

9__inference_private__transformer_block_layer_call_fn_6875

inputs
unknown:  
	unknown_0: 
	unknown_1:  
	unknown_2: 
	unknown_3:  
	unknown_4: 
	unknown_5:  
	unknown_6: 
	unknown_7: 
	unknown_8: 
	unknown_9:  

unknown_10: 

unknown_11:  

unknown_12: 

unknown_13: 

unknown_14: 
identity¢StatefulPartitionedCall©
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_private__transformer_block_layer_call_and_return_conditional_losses_5664t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:ÿÿÿÿÿÿÿÿÿ : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs

¦
D__inference_sequential_layer_call_and_return_conditional_losses_4875
dense_4_input
dense_4_4864:  
dense_4_4866: 
dense_5_4869:  
dense_5_4871: 
identity¢dense_4/StatefulPartitionedCall¢dense_5/StatefulPartitionedCallò
dense_4/StatefulPartitionedCallStatefulPartitionedCalldense_4_inputdense_4_4864dense_4_4866*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dense_4_layer_call_and_return_conditional_losses_4720
dense_5/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0dense_5_4869dense_5_4871*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dense_5_layer_call_and_return_conditional_losses_4756|
IdentityIdentity(dense_5/StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
NoOpNoOp ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ : : : : 2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall:[ W
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
'
_user_specified_namedense_4_input
¿
Ó
)__inference_sequential_layer_call_fn_4774
dense_4_input
unknown:  
	unknown_0: 
	unknown_1:  
	unknown_2: 
identity¢StatefulPartitionedCallÿ
StatefulPartitionedCallStatefulPartitionedCalldense_4_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_4763t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
'
_user_specified_namedense_4_input

D
(__inference_dropout_2_layer_call_fn_7393

inputs
identity®
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dropout_2_layer_call_and_return_conditional_losses_5209`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Ö
a
C__inference_dropout_2_layer_call_and_return_conditional_losses_5209

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ [

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs


D__inference_sequential_layer_call_and_return_conditional_losses_4763

inputs
dense_4_4721:  
dense_4_4723: 
dense_5_4757:  
dense_5_4759: 
identity¢dense_4/StatefulPartitionedCall¢dense_5/StatefulPartitionedCallë
dense_4/StatefulPartitionedCallStatefulPartitionedCallinputsdense_4_4721dense_4_4723*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dense_4_layer_call_and_return_conditional_losses_4720
dense_5/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0dense_5_4757dense_5_4759*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dense_5_layer_call_and_return_conditional_losses_4756|
IdentityIdentity(dense_5/StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
NoOpNoOp ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ : : : : 2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
ñ	
b
C__inference_dropout_2_layer_call_and_return_conditional_losses_5363

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=¦
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
¯

?__inference_model_layer_call_and_return_conditional_losses_6458

inputsZ
Gprivate__token_and_position_embedding_embedding_1_embedding_lookup_6187:	 X
Eprivate__token_and_position_embedding_embedding_embedding_lookup_6193:	Æ w
eprivate__transformer_block_private__multi_head_self_attention_dense_tensordot_readvariableop_resource:  q
cprivate__transformer_block_private__multi_head_self_attention_dense_biasadd_readvariableop_resource: y
gprivate__transformer_block_private__multi_head_self_attention_dense_1_tensordot_readvariableop_resource:  s
eprivate__transformer_block_private__multi_head_self_attention_dense_1_biasadd_readvariableop_resource: y
gprivate__transformer_block_private__multi_head_self_attention_dense_2_tensordot_readvariableop_resource:  s
eprivate__transformer_block_private__multi_head_self_attention_dense_2_biasadd_readvariableop_resource: y
gprivate__transformer_block_private__multi_head_self_attention_dense_3_tensordot_readvariableop_resource:  s
eprivate__transformer_block_private__multi_head_self_attention_dense_3_biasadd_readvariableop_resource: b
Tprivate__transformer_block_layer_normalization_batchnorm_mul_readvariableop_resource: ^
Pprivate__transformer_block_layer_normalization_batchnorm_readvariableop_resource: a
Oprivate__transformer_block_sequential_dense_4_tensordot_readvariableop_resource:  [
Mprivate__transformer_block_sequential_dense_4_biasadd_readvariableop_resource: a
Oprivate__transformer_block_sequential_dense_5_tensordot_readvariableop_resource:  [
Mprivate__transformer_block_sequential_dense_5_biasadd_readvariableop_resource: d
Vprivate__transformer_block_layer_normalization_1_batchnorm_mul_readvariableop_resource: `
Rprivate__transformer_block_layer_normalization_1_batchnorm_readvariableop_resource: 8
&dense_6_matmul_readvariableop_resource: 5
'dense_6_biasadd_readvariableop_resource:8
&dense_7_matmul_readvariableop_resource:5
'dense_7_biasadd_readvariableop_resource:
identity¢dense_6/BiasAdd/ReadVariableOp¢dense_6/MatMul/ReadVariableOp¢dense_7/BiasAdd/ReadVariableOp¢dense_7/MatMul/ReadVariableOp¢@private__token_and_position_embedding/embedding/embedding_lookup¢Bprivate__token_and_position_embedding/embedding_1/embedding_lookup¢Gprivate__transformer_block/layer_normalization/batchnorm/ReadVariableOp¢Kprivate__transformer_block/layer_normalization/batchnorm/mul/ReadVariableOp¢Iprivate__transformer_block/layer_normalization_1/batchnorm/ReadVariableOp¢Mprivate__transformer_block/layer_normalization_1/batchnorm/mul/ReadVariableOp¢Zprivate__transformer_block/private__multi_head_self_attention/dense/BiasAdd/ReadVariableOp¢\private__transformer_block/private__multi_head_self_attention/dense/Tensordot/ReadVariableOp¢\private__transformer_block/private__multi_head_self_attention/dense_1/BiasAdd/ReadVariableOp¢^private__transformer_block/private__multi_head_self_attention/dense_1/Tensordot/ReadVariableOp¢\private__transformer_block/private__multi_head_self_attention/dense_2/BiasAdd/ReadVariableOp¢^private__transformer_block/private__multi_head_self_attention/dense_2/Tensordot/ReadVariableOp¢\private__transformer_block/private__multi_head_self_attention/dense_3/BiasAdd/ReadVariableOp¢^private__transformer_block/private__multi_head_self_attention/dense_3/Tensordot/ReadVariableOp¢Dprivate__transformer_block/sequential/dense_4/BiasAdd/ReadVariableOp¢Fprivate__transformer_block/sequential/dense_4/Tensordot/ReadVariableOp¢Dprivate__transformer_block/sequential/dense_5/BiasAdd/ReadVariableOp¢Fprivate__transformer_block/sequential/dense_5/Tensordot/ReadVariableOpa
+private__token_and_position_embedding/ShapeShapeinputs*
T0*
_output_shapes
:
9private__token_and_position_embedding/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ
;private__token_and_position_embedding/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
;private__token_and_position_embedding/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
3private__token_and_position_embedding/strided_sliceStridedSlice4private__token_and_position_embedding/Shape:output:0Bprivate__token_and_position_embedding/strided_slice/stack:output:0Dprivate__token_and_position_embedding/strided_slice/stack_1:output:0Dprivate__token_and_position_embedding/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masks
1private__token_and_position_embedding/range/startConst*
_output_shapes
: *
dtype0*
value	B : s
1private__token_and_position_embedding/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :
+private__token_and_position_embedding/rangeRange:private__token_and_position_embedding/range/start:output:0<private__token_and_position_embedding/strided_slice:output:0:private__token_and_position_embedding/range/delta:output:0*
_output_shapes	
:í
Bprivate__token_and_position_embedding/embedding_1/embedding_lookupResourceGatherGprivate__token_and_position_embedding_embedding_1_embedding_lookup_61874private__token_and_position_embedding/range:output:0*
Tindices0*Z
_classP
NLloc:@private__token_and_position_embedding/embedding_1/embedding_lookup/6187*
_output_shapes
:	 *
dtype0ª
Kprivate__token_and_position_embedding/embedding_1/embedding_lookup/IdentityIdentityKprivate__token_and_position_embedding/embedding_1/embedding_lookup:output:0*
T0*Z
_classP
NLloc:@private__token_and_position_embedding/embedding_1/embedding_lookup/6187*
_output_shapes
:	 Ù
Mprivate__token_and_position_embedding/embedding_1/embedding_lookup/Identity_1IdentityTprivate__token_and_position_embedding/embedding_1/embedding_lookup/Identity:output:0*
T0*
_output_shapes
:	 
4private__token_and_position_embedding/embedding/CastCastinputs*

DstT0*

SrcT0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿø
@private__token_and_position_embedding/embedding/embedding_lookupResourceGatherEprivate__token_and_position_embedding_embedding_embedding_lookup_61938private__token_and_position_embedding/embedding/Cast:y:0*
Tindices0*X
_classN
LJloc:@private__token_and_position_embedding/embedding/embedding_lookup/6193*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
dtype0±
Iprivate__token_and_position_embedding/embedding/embedding_lookup/IdentityIdentityIprivate__token_and_position_embedding/embedding/embedding_lookup:output:0*
T0*X
_classN
LJloc:@private__token_and_position_embedding/embedding/embedding_lookup/6193*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ â
Kprivate__token_and_position_embedding/embedding/embedding_lookup/Identity_1IdentityRprivate__token_and_position_embedding/embedding/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
)private__token_and_position_embedding/addAddV2Tprivate__token_and_position_embedding/embedding/embedding_lookup/Identity_1:output:0Vprivate__token_and_position_embedding/embedding_1/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
Cprivate__transformer_block/private__multi_head_self_attention/ShapeShape-private__token_and_position_embedding/add:z:0*
T0*
_output_shapes
:
Qprivate__transformer_block/private__multi_head_self_attention/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Sprivate__transformer_block/private__multi_head_self_attention/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Sprivate__transformer_block/private__multi_head_self_attention/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
Kprivate__transformer_block/private__multi_head_self_attention/strided_sliceStridedSliceLprivate__transformer_block/private__multi_head_self_attention/Shape:output:0Zprivate__transformer_block/private__multi_head_self_attention/strided_slice/stack:output:0\private__transformer_block/private__multi_head_self_attention/strided_slice/stack_1:output:0\private__transformer_block/private__multi_head_self_attention/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
\private__transformer_block/private__multi_head_self_attention/dense/Tensordot/ReadVariableOpReadVariableOpeprivate__transformer_block_private__multi_head_self_attention_dense_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype0
Rprivate__transformer_block/private__multi_head_self_attention/dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:£
Rprivate__transformer_block/private__multi_head_self_attention/dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       °
Sprivate__transformer_block/private__multi_head_self_attention/dense/Tensordot/ShapeShape-private__token_and_position_embedding/add:z:0*
T0*
_output_shapes
:
[private__transformer_block/private__multi_head_self_attention/dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : Ë
Vprivate__transformer_block/private__multi_head_self_attention/dense/Tensordot/GatherV2GatherV2\private__transformer_block/private__multi_head_self_attention/dense/Tensordot/Shape:output:0[private__transformer_block/private__multi_head_self_attention/dense/Tensordot/free:output:0dprivate__transformer_block/private__multi_head_self_attention/dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:
]private__transformer_block/private__multi_head_self_attention/dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ï
Xprivate__transformer_block/private__multi_head_self_attention/dense/Tensordot/GatherV2_1GatherV2\private__transformer_block/private__multi_head_self_attention/dense/Tensordot/Shape:output:0[private__transformer_block/private__multi_head_self_attention/dense/Tensordot/axes:output:0fprivate__transformer_block/private__multi_head_self_attention/dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:
Sprivate__transformer_block/private__multi_head_self_attention/dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: º
Rprivate__transformer_block/private__multi_head_self_attention/dense/Tensordot/ProdProd_private__transformer_block/private__multi_head_self_attention/dense/Tensordot/GatherV2:output:0\private__transformer_block/private__multi_head_self_attention/dense/Tensordot/Const:output:0*
T0*
_output_shapes
: 
Uprivate__transformer_block/private__multi_head_self_attention/dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: À
Tprivate__transformer_block/private__multi_head_self_attention/dense/Tensordot/Prod_1Prodaprivate__transformer_block/private__multi_head_self_attention/dense/Tensordot/GatherV2_1:output:0^private__transformer_block/private__multi_head_self_attention/dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 
Yprivate__transformer_block/private__multi_head_self_attention/dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ¬
Tprivate__transformer_block/private__multi_head_self_attention/dense/Tensordot/concatConcatV2[private__transformer_block/private__multi_head_self_attention/dense/Tensordot/free:output:0[private__transformer_block/private__multi_head_self_attention/dense/Tensordot/axes:output:0bprivate__transformer_block/private__multi_head_self_attention/dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:Å
Sprivate__transformer_block/private__multi_head_self_attention/dense/Tensordot/stackPack[private__transformer_block/private__multi_head_self_attention/dense/Tensordot/Prod:output:0]private__transformer_block/private__multi_head_self_attention/dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:©
Wprivate__transformer_block/private__multi_head_self_attention/dense/Tensordot/transpose	Transpose-private__token_and_position_embedding/add:z:0]private__transformer_block/private__multi_head_self_attention/dense/Tensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ö
Uprivate__transformer_block/private__multi_head_self_attention/dense/Tensordot/ReshapeReshape[private__transformer_block/private__multi_head_self_attention/dense/Tensordot/transpose:y:0\private__transformer_block/private__multi_head_self_attention/dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÖ
Tprivate__transformer_block/private__multi_head_self_attention/dense/Tensordot/MatMulMatMul^private__transformer_block/private__multi_head_self_attention/dense/Tensordot/Reshape:output:0dprivate__transformer_block/private__multi_head_self_attention/dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
Uprivate__transformer_block/private__multi_head_self_attention/dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 
[private__transformer_block/private__multi_head_self_attention/dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ·
Vprivate__transformer_block/private__multi_head_self_attention/dense/Tensordot/concat_1ConcatV2_private__transformer_block/private__multi_head_self_attention/dense/Tensordot/GatherV2:output:0^private__transformer_block/private__multi_head_self_attention/dense/Tensordot/Const_2:output:0dprivate__transformer_block/private__multi_head_self_attention/dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:Ð
Mprivate__transformer_block/private__multi_head_self_attention/dense/TensordotReshape^private__transformer_block/private__multi_head_self_attention/dense/Tensordot/MatMul:product:0_private__transformer_block/private__multi_head_self_attention/dense/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ú
Zprivate__transformer_block/private__multi_head_self_attention/dense/BiasAdd/ReadVariableOpReadVariableOpcprivate__transformer_block_private__multi_head_self_attention_dense_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0É
Kprivate__transformer_block/private__multi_head_self_attention/dense/BiasAddBiasAddVprivate__transformer_block/private__multi_head_self_attention/dense/Tensordot:output:0bprivate__transformer_block/private__multi_head_self_attention/dense/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
^private__transformer_block/private__multi_head_self_attention/dense_1/Tensordot/ReadVariableOpReadVariableOpgprivate__transformer_block_private__multi_head_self_attention_dense_1_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype0
Tprivate__transformer_block/private__multi_head_self_attention/dense_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:¥
Tprivate__transformer_block/private__multi_head_self_attention/dense_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       ²
Uprivate__transformer_block/private__multi_head_self_attention/dense_1/Tensordot/ShapeShape-private__token_and_position_embedding/add:z:0*
T0*
_output_shapes
:
]private__transformer_block/private__multi_head_self_attention/dense_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : Ó
Xprivate__transformer_block/private__multi_head_self_attention/dense_1/Tensordot/GatherV2GatherV2^private__transformer_block/private__multi_head_self_attention/dense_1/Tensordot/Shape:output:0]private__transformer_block/private__multi_head_self_attention/dense_1/Tensordot/free:output:0fprivate__transformer_block/private__multi_head_self_attention/dense_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:¡
_private__transformer_block/private__multi_head_self_attention/dense_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ×
Zprivate__transformer_block/private__multi_head_self_attention/dense_1/Tensordot/GatherV2_1GatherV2^private__transformer_block/private__multi_head_self_attention/dense_1/Tensordot/Shape:output:0]private__transformer_block/private__multi_head_self_attention/dense_1/Tensordot/axes:output:0hprivate__transformer_block/private__multi_head_self_attention/dense_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:
Uprivate__transformer_block/private__multi_head_self_attention/dense_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: À
Tprivate__transformer_block/private__multi_head_self_attention/dense_1/Tensordot/ProdProdaprivate__transformer_block/private__multi_head_self_attention/dense_1/Tensordot/GatherV2:output:0^private__transformer_block/private__multi_head_self_attention/dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: ¡
Wprivate__transformer_block/private__multi_head_self_attention/dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: Æ
Vprivate__transformer_block/private__multi_head_self_attention/dense_1/Tensordot/Prod_1Prodcprivate__transformer_block/private__multi_head_self_attention/dense_1/Tensordot/GatherV2_1:output:0`private__transformer_block/private__multi_head_self_attention/dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 
[private__transformer_block/private__multi_head_self_attention/dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ´
Vprivate__transformer_block/private__multi_head_self_attention/dense_1/Tensordot/concatConcatV2]private__transformer_block/private__multi_head_self_attention/dense_1/Tensordot/free:output:0]private__transformer_block/private__multi_head_self_attention/dense_1/Tensordot/axes:output:0dprivate__transformer_block/private__multi_head_self_attention/dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:Ë
Uprivate__transformer_block/private__multi_head_self_attention/dense_1/Tensordot/stackPack]private__transformer_block/private__multi_head_self_attention/dense_1/Tensordot/Prod:output:0_private__transformer_block/private__multi_head_self_attention/dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:­
Yprivate__transformer_block/private__multi_head_self_attention/dense_1/Tensordot/transpose	Transpose-private__token_and_position_embedding/add:z:0_private__transformer_block/private__multi_head_self_attention/dense_1/Tensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ü
Wprivate__transformer_block/private__multi_head_self_attention/dense_1/Tensordot/ReshapeReshape]private__transformer_block/private__multi_head_self_attention/dense_1/Tensordot/transpose:y:0^private__transformer_block/private__multi_head_self_attention/dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÜ
Vprivate__transformer_block/private__multi_head_self_attention/dense_1/Tensordot/MatMulMatMul`private__transformer_block/private__multi_head_self_attention/dense_1/Tensordot/Reshape:output:0fprivate__transformer_block/private__multi_head_self_attention/dense_1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¡
Wprivate__transformer_block/private__multi_head_self_attention/dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 
]private__transformer_block/private__multi_head_self_attention/dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ¿
Xprivate__transformer_block/private__multi_head_self_attention/dense_1/Tensordot/concat_1ConcatV2aprivate__transformer_block/private__multi_head_self_attention/dense_1/Tensordot/GatherV2:output:0`private__transformer_block/private__multi_head_self_attention/dense_1/Tensordot/Const_2:output:0fprivate__transformer_block/private__multi_head_self_attention/dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:Ö
Oprivate__transformer_block/private__multi_head_self_attention/dense_1/TensordotReshape`private__transformer_block/private__multi_head_self_attention/dense_1/Tensordot/MatMul:product:0aprivate__transformer_block/private__multi_head_self_attention/dense_1/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ þ
\private__transformer_block/private__multi_head_self_attention/dense_1/BiasAdd/ReadVariableOpReadVariableOpeprivate__transformer_block_private__multi_head_self_attention_dense_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Ï
Mprivate__transformer_block/private__multi_head_self_attention/dense_1/BiasAddBiasAddXprivate__transformer_block/private__multi_head_self_attention/dense_1/Tensordot:output:0dprivate__transformer_block/private__multi_head_self_attention/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
^private__transformer_block/private__multi_head_self_attention/dense_2/Tensordot/ReadVariableOpReadVariableOpgprivate__transformer_block_private__multi_head_self_attention_dense_2_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype0
Tprivate__transformer_block/private__multi_head_self_attention/dense_2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:¥
Tprivate__transformer_block/private__multi_head_self_attention/dense_2/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       ²
Uprivate__transformer_block/private__multi_head_self_attention/dense_2/Tensordot/ShapeShape-private__token_and_position_embedding/add:z:0*
T0*
_output_shapes
:
]private__transformer_block/private__multi_head_self_attention/dense_2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : Ó
Xprivate__transformer_block/private__multi_head_self_attention/dense_2/Tensordot/GatherV2GatherV2^private__transformer_block/private__multi_head_self_attention/dense_2/Tensordot/Shape:output:0]private__transformer_block/private__multi_head_self_attention/dense_2/Tensordot/free:output:0fprivate__transformer_block/private__multi_head_self_attention/dense_2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:¡
_private__transformer_block/private__multi_head_self_attention/dense_2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ×
Zprivate__transformer_block/private__multi_head_self_attention/dense_2/Tensordot/GatherV2_1GatherV2^private__transformer_block/private__multi_head_self_attention/dense_2/Tensordot/Shape:output:0]private__transformer_block/private__multi_head_self_attention/dense_2/Tensordot/axes:output:0hprivate__transformer_block/private__multi_head_self_attention/dense_2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:
Uprivate__transformer_block/private__multi_head_self_attention/dense_2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: À
Tprivate__transformer_block/private__multi_head_self_attention/dense_2/Tensordot/ProdProdaprivate__transformer_block/private__multi_head_self_attention/dense_2/Tensordot/GatherV2:output:0^private__transformer_block/private__multi_head_self_attention/dense_2/Tensordot/Const:output:0*
T0*
_output_shapes
: ¡
Wprivate__transformer_block/private__multi_head_self_attention/dense_2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: Æ
Vprivate__transformer_block/private__multi_head_self_attention/dense_2/Tensordot/Prod_1Prodcprivate__transformer_block/private__multi_head_self_attention/dense_2/Tensordot/GatherV2_1:output:0`private__transformer_block/private__multi_head_self_attention/dense_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 
[private__transformer_block/private__multi_head_self_attention/dense_2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ´
Vprivate__transformer_block/private__multi_head_self_attention/dense_2/Tensordot/concatConcatV2]private__transformer_block/private__multi_head_self_attention/dense_2/Tensordot/free:output:0]private__transformer_block/private__multi_head_self_attention/dense_2/Tensordot/axes:output:0dprivate__transformer_block/private__multi_head_self_attention/dense_2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:Ë
Uprivate__transformer_block/private__multi_head_self_attention/dense_2/Tensordot/stackPack]private__transformer_block/private__multi_head_self_attention/dense_2/Tensordot/Prod:output:0_private__transformer_block/private__multi_head_self_attention/dense_2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:­
Yprivate__transformer_block/private__multi_head_self_attention/dense_2/Tensordot/transpose	Transpose-private__token_and_position_embedding/add:z:0_private__transformer_block/private__multi_head_self_attention/dense_2/Tensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ü
Wprivate__transformer_block/private__multi_head_self_attention/dense_2/Tensordot/ReshapeReshape]private__transformer_block/private__multi_head_self_attention/dense_2/Tensordot/transpose:y:0^private__transformer_block/private__multi_head_self_attention/dense_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÜ
Vprivate__transformer_block/private__multi_head_self_attention/dense_2/Tensordot/MatMulMatMul`private__transformer_block/private__multi_head_self_attention/dense_2/Tensordot/Reshape:output:0fprivate__transformer_block/private__multi_head_self_attention/dense_2/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¡
Wprivate__transformer_block/private__multi_head_self_attention/dense_2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 
]private__transformer_block/private__multi_head_self_attention/dense_2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ¿
Xprivate__transformer_block/private__multi_head_self_attention/dense_2/Tensordot/concat_1ConcatV2aprivate__transformer_block/private__multi_head_self_attention/dense_2/Tensordot/GatherV2:output:0`private__transformer_block/private__multi_head_self_attention/dense_2/Tensordot/Const_2:output:0fprivate__transformer_block/private__multi_head_self_attention/dense_2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:Ö
Oprivate__transformer_block/private__multi_head_self_attention/dense_2/TensordotReshape`private__transformer_block/private__multi_head_self_attention/dense_2/Tensordot/MatMul:product:0aprivate__transformer_block/private__multi_head_self_attention/dense_2/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ þ
\private__transformer_block/private__multi_head_self_attention/dense_2/BiasAdd/ReadVariableOpReadVariableOpeprivate__transformer_block_private__multi_head_self_attention_dense_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Ï
Mprivate__transformer_block/private__multi_head_self_attention/dense_2/BiasAddBiasAddXprivate__transformer_block/private__multi_head_self_attention/dense_2/Tensordot:output:0dprivate__transformer_block/private__multi_head_self_attention/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
Mprivate__transformer_block/private__multi_head_self_attention/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
Mprivate__transformer_block/private__multi_head_self_attention/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :
Mprivate__transformer_block/private__multi_head_self_attention/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :ß
Kprivate__transformer_block/private__multi_head_self_attention/Reshape/shapePackTprivate__transformer_block/private__multi_head_self_attention/strided_slice:output:0Vprivate__transformer_block/private__multi_head_self_attention/Reshape/shape/1:output:0Vprivate__transformer_block/private__multi_head_self_attention/Reshape/shape/2:output:0Vprivate__transformer_block/private__multi_head_self_attention/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:¿
Eprivate__transformer_block/private__multi_head_self_attention/ReshapeReshapeTprivate__transformer_block/private__multi_head_self_attention/dense/BiasAdd:output:0Tprivate__transformer_block/private__multi_head_self_attention/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¥
Lprivate__transformer_block/private__multi_head_self_attention/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             ¾
Gprivate__transformer_block/private__multi_head_self_attention/transpose	TransposeNprivate__transformer_block/private__multi_head_self_attention/Reshape:output:0Uprivate__transformer_block/private__multi_head_self_attention/transpose/perm:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Oprivate__transformer_block/private__multi_head_self_attention/Reshape_1/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
Oprivate__transformer_block/private__multi_head_self_attention/Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :
Oprivate__transformer_block/private__multi_head_self_attention/Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :ç
Mprivate__transformer_block/private__multi_head_self_attention/Reshape_1/shapePackTprivate__transformer_block/private__multi_head_self_attention/strided_slice:output:0Xprivate__transformer_block/private__multi_head_self_attention/Reshape_1/shape/1:output:0Xprivate__transformer_block/private__multi_head_self_attention/Reshape_1/shape/2:output:0Xprivate__transformer_block/private__multi_head_self_attention/Reshape_1/shape/3:output:0*
N*
T0*
_output_shapes
:Å
Gprivate__transformer_block/private__multi_head_self_attention/Reshape_1ReshapeVprivate__transformer_block/private__multi_head_self_attention/dense_1/BiasAdd:output:0Vprivate__transformer_block/private__multi_head_self_attention/Reshape_1/shape:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ§
Nprivate__transformer_block/private__multi_head_self_attention/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             Ä
Iprivate__transformer_block/private__multi_head_self_attention/transpose_1	TransposePprivate__transformer_block/private__multi_head_self_attention/Reshape_1:output:0Wprivate__transformer_block/private__multi_head_self_attention/transpose_1/perm:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Oprivate__transformer_block/private__multi_head_self_attention/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
Oprivate__transformer_block/private__multi_head_self_attention/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :
Oprivate__transformer_block/private__multi_head_self_attention/Reshape_2/shape/3Const*
_output_shapes
: *
dtype0*
value	B :ç
Mprivate__transformer_block/private__multi_head_self_attention/Reshape_2/shapePackTprivate__transformer_block/private__multi_head_self_attention/strided_slice:output:0Xprivate__transformer_block/private__multi_head_self_attention/Reshape_2/shape/1:output:0Xprivate__transformer_block/private__multi_head_self_attention/Reshape_2/shape/2:output:0Xprivate__transformer_block/private__multi_head_self_attention/Reshape_2/shape/3:output:0*
N*
T0*
_output_shapes
:Å
Gprivate__transformer_block/private__multi_head_self_attention/Reshape_2ReshapeVprivate__transformer_block/private__multi_head_self_attention/dense_2/BiasAdd:output:0Vprivate__transformer_block/private__multi_head_self_attention/Reshape_2/shape:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ§
Nprivate__transformer_block/private__multi_head_self_attention/transpose_2/permConst*
_output_shapes
:*
dtype0*%
valueB"             Ä
Iprivate__transformer_block/private__multi_head_self_attention/transpose_2	TransposePprivate__transformer_block/private__multi_head_self_attention/Reshape_2:output:0Wprivate__transformer_block/private__multi_head_self_attention/transpose_2/perm:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÊ
Dprivate__transformer_block/private__multi_head_self_attention/MatMulBatchMatMulV2Kprivate__transformer_block/private__multi_head_self_attention/transpose:y:0Mprivate__transformer_block/private__multi_head_self_attention/transpose_1:y:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
adj_y(Â
Eprivate__transformer_block/private__multi_head_self_attention/Shape_1ShapeMprivate__transformer_block/private__multi_head_self_attention/transpose_1:y:0*
T0*
_output_shapes
:¦
Sprivate__transformer_block/private__multi_head_self_attention/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ
Uprivate__transformer_block/private__multi_head_self_attention/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
Uprivate__transformer_block/private__multi_head_self_attention/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
Mprivate__transformer_block/private__multi_head_self_attention/strided_slice_1StridedSliceNprivate__transformer_block/private__multi_head_self_attention/Shape_1:output:0\private__transformer_block/private__multi_head_self_attention/strided_slice_1/stack:output:0^private__transformer_block/private__multi_head_self_attention/strided_slice_1/stack_1:output:0^private__transformer_block/private__multi_head_self_attention/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskÒ
Bprivate__transformer_block/private__multi_head_self_attention/CastCastVprivate__transformer_block/private__multi_head_self_attention/strided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: ³
Bprivate__transformer_block/private__multi_head_self_attention/SqrtSqrtFprivate__transformer_block/private__multi_head_self_attention/Cast:y:0*
T0*
_output_shapes
: ³
Eprivate__transformer_block/private__multi_head_self_attention/truedivRealDivMprivate__transformer_block/private__multi_head_self_attention/MatMul:output:0Fprivate__transformer_block/private__multi_head_self_attention/Sqrt:y:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿç
Eprivate__transformer_block/private__multi_head_self_attention/SoftmaxSoftmaxIprivate__transformer_block/private__multi_head_self_attention/truediv:z:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿº
Fprivate__transformer_block/private__multi_head_self_attention/MatMul_1BatchMatMulV2Oprivate__transformer_block/private__multi_head_self_attention/Softmax:softmax:0Mprivate__transformer_block/private__multi_head_self_attention/transpose_2:y:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ§
Nprivate__transformer_block/private__multi_head_self_attention/transpose_3/permConst*
_output_shapes
:*
dtype0*%
valueB"             Ã
Iprivate__transformer_block/private__multi_head_self_attention/transpose_3	TransposeOprivate__transformer_block/private__multi_head_self_attention/MatMul_1:output:0Wprivate__transformer_block/private__multi_head_self_attention/transpose_3/perm:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Oprivate__transformer_block/private__multi_head_self_attention/Reshape_3/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
Oprivate__transformer_block/private__multi_head_self_attention/Reshape_3/shape/2Const*
_output_shapes
: *
dtype0*
value	B : 
Mprivate__transformer_block/private__multi_head_self_attention/Reshape_3/shapePackTprivate__transformer_block/private__multi_head_self_attention/strided_slice:output:0Xprivate__transformer_block/private__multi_head_self_attention/Reshape_3/shape/1:output:0Xprivate__transformer_block/private__multi_head_self_attention/Reshape_3/shape/2:output:0*
N*
T0*
_output_shapes
:¸
Gprivate__transformer_block/private__multi_head_self_attention/Reshape_3ReshapeMprivate__transformer_block/private__multi_head_self_attention/transpose_3:y:0Vprivate__transformer_block/private__multi_head_self_attention/Reshape_3/shape:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
^private__transformer_block/private__multi_head_self_attention/dense_3/Tensordot/ReadVariableOpReadVariableOpgprivate__transformer_block_private__multi_head_self_attention_dense_3_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype0
Tprivate__transformer_block/private__multi_head_self_attention/dense_3/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:¥
Tprivate__transformer_block/private__multi_head_self_attention/dense_3/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       Õ
Uprivate__transformer_block/private__multi_head_self_attention/dense_3/Tensordot/ShapeShapePprivate__transformer_block/private__multi_head_self_attention/Reshape_3:output:0*
T0*
_output_shapes
:
]private__transformer_block/private__multi_head_self_attention/dense_3/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : Ó
Xprivate__transformer_block/private__multi_head_self_attention/dense_3/Tensordot/GatherV2GatherV2^private__transformer_block/private__multi_head_self_attention/dense_3/Tensordot/Shape:output:0]private__transformer_block/private__multi_head_self_attention/dense_3/Tensordot/free:output:0fprivate__transformer_block/private__multi_head_self_attention/dense_3/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:¡
_private__transformer_block/private__multi_head_self_attention/dense_3/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ×
Zprivate__transformer_block/private__multi_head_self_attention/dense_3/Tensordot/GatherV2_1GatherV2^private__transformer_block/private__multi_head_self_attention/dense_3/Tensordot/Shape:output:0]private__transformer_block/private__multi_head_self_attention/dense_3/Tensordot/axes:output:0hprivate__transformer_block/private__multi_head_self_attention/dense_3/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:
Uprivate__transformer_block/private__multi_head_self_attention/dense_3/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: À
Tprivate__transformer_block/private__multi_head_self_attention/dense_3/Tensordot/ProdProdaprivate__transformer_block/private__multi_head_self_attention/dense_3/Tensordot/GatherV2:output:0^private__transformer_block/private__multi_head_self_attention/dense_3/Tensordot/Const:output:0*
T0*
_output_shapes
: ¡
Wprivate__transformer_block/private__multi_head_self_attention/dense_3/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: Æ
Vprivate__transformer_block/private__multi_head_self_attention/dense_3/Tensordot/Prod_1Prodcprivate__transformer_block/private__multi_head_self_attention/dense_3/Tensordot/GatherV2_1:output:0`private__transformer_block/private__multi_head_self_attention/dense_3/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 
[private__transformer_block/private__multi_head_self_attention/dense_3/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ´
Vprivate__transformer_block/private__multi_head_self_attention/dense_3/Tensordot/concatConcatV2]private__transformer_block/private__multi_head_self_attention/dense_3/Tensordot/free:output:0]private__transformer_block/private__multi_head_self_attention/dense_3/Tensordot/axes:output:0dprivate__transformer_block/private__multi_head_self_attention/dense_3/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:Ë
Uprivate__transformer_block/private__multi_head_self_attention/dense_3/Tensordot/stackPack]private__transformer_block/private__multi_head_self_attention/dense_3/Tensordot/Prod:output:0_private__transformer_block/private__multi_head_self_attention/dense_3/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:Ø
Yprivate__transformer_block/private__multi_head_self_attention/dense_3/Tensordot/transpose	TransposePprivate__transformer_block/private__multi_head_self_attention/Reshape_3:output:0_private__transformer_block/private__multi_head_self_attention/dense_3/Tensordot/concat:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ Ü
Wprivate__transformer_block/private__multi_head_self_attention/dense_3/Tensordot/ReshapeReshape]private__transformer_block/private__multi_head_self_attention/dense_3/Tensordot/transpose:y:0^private__transformer_block/private__multi_head_self_attention/dense_3/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÜ
Vprivate__transformer_block/private__multi_head_self_attention/dense_3/Tensordot/MatMulMatMul`private__transformer_block/private__multi_head_self_attention/dense_3/Tensordot/Reshape:output:0fprivate__transformer_block/private__multi_head_self_attention/dense_3/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¡
Wprivate__transformer_block/private__multi_head_self_attention/dense_3/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 
]private__transformer_block/private__multi_head_self_attention/dense_3/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ¿
Xprivate__transformer_block/private__multi_head_self_attention/dense_3/Tensordot/concat_1ConcatV2aprivate__transformer_block/private__multi_head_self_attention/dense_3/Tensordot/GatherV2:output:0`private__transformer_block/private__multi_head_self_attention/dense_3/Tensordot/Const_2:output:0fprivate__transformer_block/private__multi_head_self_attention/dense_3/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:Þ
Oprivate__transformer_block/private__multi_head_self_attention/dense_3/TensordotReshape`private__transformer_block/private__multi_head_self_attention/dense_3/Tensordot/MatMul:product:0aprivate__transformer_block/private__multi_head_self_attention/dense_3/Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ þ
\private__transformer_block/private__multi_head_self_attention/dense_3/BiasAdd/ReadVariableOpReadVariableOpeprivate__transformer_block_private__multi_head_self_attention_dense_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0×
Mprivate__transformer_block/private__multi_head_self_attention/dense_3/BiasAddBiasAddXprivate__transformer_block/private__multi_head_self_attention/dense_3/Tensordot:output:0dprivate__transformer_block/private__multi_head_self_attention/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ Î
+private__transformer_block/dropout/IdentityIdentityVprivate__transformer_block/private__multi_head_self_attention/dense_3/BiasAdd:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ Ã
private__transformer_block/addAddV2-private__token_and_position_embedding/add:z:04private__transformer_block/dropout/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
Mprivate__transformer_block/layer_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
;private__transformer_block/layer_normalization/moments/meanMean"private__transformer_block/add:z:0Vprivate__transformer_block/layer_normalization/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(Ð
Cprivate__transformer_block/layer_normalization/moments/StopGradientStopGradientDprivate__transformer_block/layer_normalization/moments/mean:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Hprivate__transformer_block/layer_normalization/moments/SquaredDifferenceSquaredDifference"private__transformer_block/add:z:0Lprivate__transformer_block/layer_normalization/moments/StopGradient:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
Qprivate__transformer_block/layer_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:¹
?private__transformer_block/layer_normalization/moments/varianceMeanLprivate__transformer_block/layer_normalization/moments/SquaredDifference:z:0Zprivate__transformer_block/layer_normalization/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(
>private__transformer_block/layer_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½75
<private__transformer_block/layer_normalization/batchnorm/addAddV2Hprivate__transformer_block/layer_normalization/moments/variance:output:0Gprivate__transformer_block/layer_normalization/batchnorm/add/y:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
>private__transformer_block/layer_normalization/batchnorm/RsqrtRsqrt@private__transformer_block/layer_normalization/batchnorm/add:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ
Kprivate__transformer_block/layer_normalization/batchnorm/mul/ReadVariableOpReadVariableOpTprivate__transformer_block_layer_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype0
<private__transformer_block/layer_normalization/batchnorm/mulMulBprivate__transformer_block/layer_normalization/batchnorm/Rsqrt:y:0Sprivate__transformer_block/layer_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ â
>private__transformer_block/layer_normalization/batchnorm/mul_1Mul"private__transformer_block/add:z:0@private__transformer_block/layer_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
>private__transformer_block/layer_normalization/batchnorm/mul_2MulDprivate__transformer_block/layer_normalization/moments/mean:output:0@private__transformer_block/layer_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ô
Gprivate__transformer_block/layer_normalization/batchnorm/ReadVariableOpReadVariableOpPprivate__transformer_block_layer_normalization_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype0
<private__transformer_block/layer_normalization/batchnorm/subSubOprivate__transformer_block/layer_normalization/batchnorm/ReadVariableOp:value:0Bprivate__transformer_block/layer_normalization/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
>private__transformer_block/layer_normalization/batchnorm/add_1AddV2Bprivate__transformer_block/layer_normalization/batchnorm/mul_1:z:0@private__transformer_block/layer_normalization/batchnorm/sub:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ö
Fprivate__transformer_block/sequential/dense_4/Tensordot/ReadVariableOpReadVariableOpOprivate__transformer_block_sequential_dense_4_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype0
<private__transformer_block/sequential/dense_4/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:
<private__transformer_block/sequential/dense_4/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       ¯
=private__transformer_block/sequential/dense_4/Tensordot/ShapeShapeBprivate__transformer_block/layer_normalization/batchnorm/add_1:z:0*
T0*
_output_shapes
:
Eprivate__transformer_block/sequential/dense_4/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ó
@private__transformer_block/sequential/dense_4/Tensordot/GatherV2GatherV2Fprivate__transformer_block/sequential/dense_4/Tensordot/Shape:output:0Eprivate__transformer_block/sequential/dense_4/Tensordot/free:output:0Nprivate__transformer_block/sequential/dense_4/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:
Gprivate__transformer_block/sequential/dense_4/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ÷
Bprivate__transformer_block/sequential/dense_4/Tensordot/GatherV2_1GatherV2Fprivate__transformer_block/sequential/dense_4/Tensordot/Shape:output:0Eprivate__transformer_block/sequential/dense_4/Tensordot/axes:output:0Pprivate__transformer_block/sequential/dense_4/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:
=private__transformer_block/sequential/dense_4/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ø
<private__transformer_block/sequential/dense_4/Tensordot/ProdProdIprivate__transformer_block/sequential/dense_4/Tensordot/GatherV2:output:0Fprivate__transformer_block/sequential/dense_4/Tensordot/Const:output:0*
T0*
_output_shapes
: 
?private__transformer_block/sequential/dense_4/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: þ
>private__transformer_block/sequential/dense_4/Tensordot/Prod_1ProdKprivate__transformer_block/sequential/dense_4/Tensordot/GatherV2_1:output:0Hprivate__transformer_block/sequential/dense_4/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 
Cprivate__transformer_block/sequential/dense_4/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ô
>private__transformer_block/sequential/dense_4/Tensordot/concatConcatV2Eprivate__transformer_block/sequential/dense_4/Tensordot/free:output:0Eprivate__transformer_block/sequential/dense_4/Tensordot/axes:output:0Lprivate__transformer_block/sequential/dense_4/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:
=private__transformer_block/sequential/dense_4/Tensordot/stackPackEprivate__transformer_block/sequential/dense_4/Tensordot/Prod:output:0Gprivate__transformer_block/sequential/dense_4/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:
Aprivate__transformer_block/sequential/dense_4/Tensordot/transpose	TransposeBprivate__transformer_block/layer_normalization/batchnorm/add_1:z:0Gprivate__transformer_block/sequential/dense_4/Tensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
?private__transformer_block/sequential/dense_4/Tensordot/ReshapeReshapeEprivate__transformer_block/sequential/dense_4/Tensordot/transpose:y:0Fprivate__transformer_block/sequential/dense_4/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
>private__transformer_block/sequential/dense_4/Tensordot/MatMulMatMulHprivate__transformer_block/sequential/dense_4/Tensordot/Reshape:output:0Nprivate__transformer_block/sequential/dense_4/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
?private__transformer_block/sequential/dense_4/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 
Eprivate__transformer_block/sequential/dense_4/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ß
@private__transformer_block/sequential/dense_4/Tensordot/concat_1ConcatV2Iprivate__transformer_block/sequential/dense_4/Tensordot/GatherV2:output:0Hprivate__transformer_block/sequential/dense_4/Tensordot/Const_2:output:0Nprivate__transformer_block/sequential/dense_4/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
7private__transformer_block/sequential/dense_4/TensordotReshapeHprivate__transformer_block/sequential/dense_4/Tensordot/MatMul:product:0Iprivate__transformer_block/sequential/dense_4/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Î
Dprivate__transformer_block/sequential/dense_4/BiasAdd/ReadVariableOpReadVariableOpMprivate__transformer_block_sequential_dense_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
5private__transformer_block/sequential/dense_4/BiasAddBiasAdd@private__transformer_block/sequential/dense_4/Tensordot:output:0Lprivate__transformer_block/sequential/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ±
2private__transformer_block/sequential/dense_4/ReluRelu>private__transformer_block/sequential/dense_4/BiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ö
Fprivate__transformer_block/sequential/dense_5/Tensordot/ReadVariableOpReadVariableOpOprivate__transformer_block_sequential_dense_5_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype0
<private__transformer_block/sequential/dense_5/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:
<private__transformer_block/sequential/dense_5/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       ­
=private__transformer_block/sequential/dense_5/Tensordot/ShapeShape@private__transformer_block/sequential/dense_4/Relu:activations:0*
T0*
_output_shapes
:
Eprivate__transformer_block/sequential/dense_5/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ó
@private__transformer_block/sequential/dense_5/Tensordot/GatherV2GatherV2Fprivate__transformer_block/sequential/dense_5/Tensordot/Shape:output:0Eprivate__transformer_block/sequential/dense_5/Tensordot/free:output:0Nprivate__transformer_block/sequential/dense_5/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:
Gprivate__transformer_block/sequential/dense_5/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ÷
Bprivate__transformer_block/sequential/dense_5/Tensordot/GatherV2_1GatherV2Fprivate__transformer_block/sequential/dense_5/Tensordot/Shape:output:0Eprivate__transformer_block/sequential/dense_5/Tensordot/axes:output:0Pprivate__transformer_block/sequential/dense_5/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:
=private__transformer_block/sequential/dense_5/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ø
<private__transformer_block/sequential/dense_5/Tensordot/ProdProdIprivate__transformer_block/sequential/dense_5/Tensordot/GatherV2:output:0Fprivate__transformer_block/sequential/dense_5/Tensordot/Const:output:0*
T0*
_output_shapes
: 
?private__transformer_block/sequential/dense_5/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: þ
>private__transformer_block/sequential/dense_5/Tensordot/Prod_1ProdKprivate__transformer_block/sequential/dense_5/Tensordot/GatherV2_1:output:0Hprivate__transformer_block/sequential/dense_5/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 
Cprivate__transformer_block/sequential/dense_5/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ô
>private__transformer_block/sequential/dense_5/Tensordot/concatConcatV2Eprivate__transformer_block/sequential/dense_5/Tensordot/free:output:0Eprivate__transformer_block/sequential/dense_5/Tensordot/axes:output:0Lprivate__transformer_block/sequential/dense_5/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:
=private__transformer_block/sequential/dense_5/Tensordot/stackPackEprivate__transformer_block/sequential/dense_5/Tensordot/Prod:output:0Gprivate__transformer_block/sequential/dense_5/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:
Aprivate__transformer_block/sequential/dense_5/Tensordot/transpose	Transpose@private__transformer_block/sequential/dense_4/Relu:activations:0Gprivate__transformer_block/sequential/dense_5/Tensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
?private__transformer_block/sequential/dense_5/Tensordot/ReshapeReshapeEprivate__transformer_block/sequential/dense_5/Tensordot/transpose:y:0Fprivate__transformer_block/sequential/dense_5/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
>private__transformer_block/sequential/dense_5/Tensordot/MatMulMatMulHprivate__transformer_block/sequential/dense_5/Tensordot/Reshape:output:0Nprivate__transformer_block/sequential/dense_5/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
?private__transformer_block/sequential/dense_5/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 
Eprivate__transformer_block/sequential/dense_5/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ß
@private__transformer_block/sequential/dense_5/Tensordot/concat_1ConcatV2Iprivate__transformer_block/sequential/dense_5/Tensordot/GatherV2:output:0Hprivate__transformer_block/sequential/dense_5/Tensordot/Const_2:output:0Nprivate__transformer_block/sequential/dense_5/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
7private__transformer_block/sequential/dense_5/TensordotReshapeHprivate__transformer_block/sequential/dense_5/Tensordot/MatMul:product:0Iprivate__transformer_block/sequential/dense_5/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Î
Dprivate__transformer_block/sequential/dense_5/BiasAdd/ReadVariableOpReadVariableOpMprivate__transformer_block_sequential_dense_5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
5private__transformer_block/sequential/dense_5/BiasAddBiasAdd@private__transformer_block/sequential/dense_5/Tensordot:output:0Lprivate__transformer_block/sequential/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ °
-private__transformer_block/dropout_1/IdentityIdentity>private__transformer_block/sequential/dense_5/BiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ü
 private__transformer_block/add_1AddV2Bprivate__transformer_block/layer_normalization/batchnorm/add_1:z:06private__transformer_block/dropout_1/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
Oprivate__transformer_block/layer_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
=private__transformer_block/layer_normalization_1/moments/meanMean$private__transformer_block/add_1:z:0Xprivate__transformer_block/layer_normalization_1/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(Ô
Eprivate__transformer_block/layer_normalization_1/moments/StopGradientStopGradientFprivate__transformer_block/layer_normalization_1/moments/mean:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Jprivate__transformer_block/layer_normalization_1/moments/SquaredDifferenceSquaredDifference$private__transformer_block/add_1:z:0Nprivate__transformer_block/layer_normalization_1/moments/StopGradient:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
Sprivate__transformer_block/layer_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:¿
Aprivate__transformer_block/layer_normalization_1/moments/varianceMeanNprivate__transformer_block/layer_normalization_1/moments/SquaredDifference:z:0\private__transformer_block/layer_normalization_1/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(
@private__transformer_block/layer_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½75
>private__transformer_block/layer_normalization_1/batchnorm/addAddV2Jprivate__transformer_block/layer_normalization_1/moments/variance:output:0Iprivate__transformer_block/layer_normalization_1/batchnorm/add/y:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄ
@private__transformer_block/layer_normalization_1/batchnorm/RsqrtRsqrtBprivate__transformer_block/layer_normalization_1/batchnorm/add:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿà
Mprivate__transformer_block/layer_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOpVprivate__transformer_block_layer_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype0
>private__transformer_block/layer_normalization_1/batchnorm/mulMulDprivate__transformer_block/layer_normalization_1/batchnorm/Rsqrt:y:0Uprivate__transformer_block/layer_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ è
@private__transformer_block/layer_normalization_1/batchnorm/mul_1Mul$private__transformer_block/add_1:z:0Bprivate__transformer_block/layer_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
@private__transformer_block/layer_normalization_1/batchnorm/mul_2MulFprivate__transformer_block/layer_normalization_1/moments/mean:output:0Bprivate__transformer_block/layer_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ø
Iprivate__transformer_block/layer_normalization_1/batchnorm/ReadVariableOpReadVariableOpRprivate__transformer_block_layer_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype0
>private__transformer_block/layer_normalization_1/batchnorm/subSubQprivate__transformer_block/layer_normalization_1/batchnorm/ReadVariableOp:value:0Dprivate__transformer_block/layer_normalization_1/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
@private__transformer_block/layer_normalization_1/batchnorm/add_1AddV2Dprivate__transformer_block/layer_normalization_1/batchnorm/mul_1:z:0Bprivate__transformer_block/layer_normalization_1/batchnorm/sub:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ q
/global_average_pooling1d/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :×
global_average_pooling1d/MeanMeanDprivate__transformer_block/layer_normalization_1/batchnorm/add_1:z:08global_average_pooling1d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ x
dropout_2/IdentityIdentity&global_average_pooling1d/Mean:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource*
_output_shapes

: *
dtype0
dense_6/MatMulMatMuldropout_2/Identity:output:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
dense_6/ReluReludense_6/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
dropout_3/IdentityIdentitydense_6/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_7/MatMul/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_7/MatMulMatMuldropout_3/Identity:output:0%dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_7/BiasAddBiasAdddense_7/MatMul:product:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
dense_7/SoftmaxSoftmaxdense_7/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
IdentityIdentitydense_7/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
NoOpNoOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp^dense_7/BiasAdd/ReadVariableOp^dense_7/MatMul/ReadVariableOpA^private__token_and_position_embedding/embedding/embedding_lookupC^private__token_and_position_embedding/embedding_1/embedding_lookupH^private__transformer_block/layer_normalization/batchnorm/ReadVariableOpL^private__transformer_block/layer_normalization/batchnorm/mul/ReadVariableOpJ^private__transformer_block/layer_normalization_1/batchnorm/ReadVariableOpN^private__transformer_block/layer_normalization_1/batchnorm/mul/ReadVariableOp[^private__transformer_block/private__multi_head_self_attention/dense/BiasAdd/ReadVariableOp]^private__transformer_block/private__multi_head_self_attention/dense/Tensordot/ReadVariableOp]^private__transformer_block/private__multi_head_self_attention/dense_1/BiasAdd/ReadVariableOp_^private__transformer_block/private__multi_head_self_attention/dense_1/Tensordot/ReadVariableOp]^private__transformer_block/private__multi_head_self_attention/dense_2/BiasAdd/ReadVariableOp_^private__transformer_block/private__multi_head_self_attention/dense_2/Tensordot/ReadVariableOp]^private__transformer_block/private__multi_head_self_attention/dense_3/BiasAdd/ReadVariableOp_^private__transformer_block/private__multi_head_self_attention/dense_3/Tensordot/ReadVariableOpE^private__transformer_block/sequential/dense_4/BiasAdd/ReadVariableOpG^private__transformer_block/sequential/dense_4/Tensordot/ReadVariableOpE^private__transformer_block/sequential/dense_5/BiasAdd/ReadVariableOpG^private__transformer_block/sequential/dense_5/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : 2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp2@
dense_7/BiasAdd/ReadVariableOpdense_7/BiasAdd/ReadVariableOp2>
dense_7/MatMul/ReadVariableOpdense_7/MatMul/ReadVariableOp2
@private__token_and_position_embedding/embedding/embedding_lookup@private__token_and_position_embedding/embedding/embedding_lookup2
Bprivate__token_and_position_embedding/embedding_1/embedding_lookupBprivate__token_and_position_embedding/embedding_1/embedding_lookup2
Gprivate__transformer_block/layer_normalization/batchnorm/ReadVariableOpGprivate__transformer_block/layer_normalization/batchnorm/ReadVariableOp2
Kprivate__transformer_block/layer_normalization/batchnorm/mul/ReadVariableOpKprivate__transformer_block/layer_normalization/batchnorm/mul/ReadVariableOp2
Iprivate__transformer_block/layer_normalization_1/batchnorm/ReadVariableOpIprivate__transformer_block/layer_normalization_1/batchnorm/ReadVariableOp2
Mprivate__transformer_block/layer_normalization_1/batchnorm/mul/ReadVariableOpMprivate__transformer_block/layer_normalization_1/batchnorm/mul/ReadVariableOp2¸
Zprivate__transformer_block/private__multi_head_self_attention/dense/BiasAdd/ReadVariableOpZprivate__transformer_block/private__multi_head_self_attention/dense/BiasAdd/ReadVariableOp2¼
\private__transformer_block/private__multi_head_self_attention/dense/Tensordot/ReadVariableOp\private__transformer_block/private__multi_head_self_attention/dense/Tensordot/ReadVariableOp2¼
\private__transformer_block/private__multi_head_self_attention/dense_1/BiasAdd/ReadVariableOp\private__transformer_block/private__multi_head_self_attention/dense_1/BiasAdd/ReadVariableOp2À
^private__transformer_block/private__multi_head_self_attention/dense_1/Tensordot/ReadVariableOp^private__transformer_block/private__multi_head_self_attention/dense_1/Tensordot/ReadVariableOp2¼
\private__transformer_block/private__multi_head_self_attention/dense_2/BiasAdd/ReadVariableOp\private__transformer_block/private__multi_head_self_attention/dense_2/BiasAdd/ReadVariableOp2À
^private__transformer_block/private__multi_head_self_attention/dense_2/Tensordot/ReadVariableOp^private__transformer_block/private__multi_head_self_attention/dense_2/Tensordot/ReadVariableOp2¼
\private__transformer_block/private__multi_head_self_attention/dense_3/BiasAdd/ReadVariableOp\private__transformer_block/private__multi_head_self_attention/dense_3/BiasAdd/ReadVariableOp2À
^private__transformer_block/private__multi_head_self_attention/dense_3/Tensordot/ReadVariableOp^private__transformer_block/private__multi_head_self_attention/dense_3/Tensordot/ReadVariableOp2
Dprivate__transformer_block/sequential/dense_4/BiasAdd/ReadVariableOpDprivate__transformer_block/sequential/dense_4/BiasAdd/ReadVariableOp2
Fprivate__transformer_block/sequential/dense_4/Tensordot/ReadVariableOpFprivate__transformer_block/sequential/dense_4/Tensordot/ReadVariableOp2
Dprivate__transformer_block/sequential/dense_5/BiasAdd/ReadVariableOpDprivate__transformer_block/sequential/dense_5/BiasAdd/ReadVariableOp2
Fprivate__transformer_block/sequential/dense_5/Tensordot/ReadVariableOpFprivate__transformer_block/sequential/dense_5/Tensordot/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

¶
$__inference_model_layer_call_fn_6176

inputs
unknown:	 
	unknown_0:	Æ 
	unknown_1:  
	unknown_2: 
	unknown_3:  
	unknown_4: 
	unknown_5:  
	unknown_6: 
	unknown_7:  
	unknown_8: 
	unknown_9: 

unknown_10: 

unknown_11:  

unknown_12: 

unknown_13:  

unknown_14: 

unknown_15: 

unknown_16: 

unknown_17: 

unknown_18:

unknown_19:

unknown_20:
identity¢StatefulPartitionedCallã
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*8
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_5815o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
í
a
(__inference_dropout_3_layer_call_fn_7445

inputs
identity¢StatefulPartitionedCall¾
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dropout_3_layer_call_and_return_conditional_losses_5330o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ø>
Ó
D__inference_sequential_layer_call_and_return_conditional_losses_7565

inputs;
)dense_4_tensordot_readvariableop_resource:  5
'dense_4_biasadd_readvariableop_resource: ;
)dense_5_tensordot_readvariableop_resource:  5
'dense_5_biasadd_readvariableop_resource: 
identity¢dense_4/BiasAdd/ReadVariableOp¢ dense_4/Tensordot/ReadVariableOp¢dense_5/BiasAdd/ReadVariableOp¢ dense_5/Tensordot/ReadVariableOp
 dense_4/Tensordot/ReadVariableOpReadVariableOp)dense_4_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype0`
dense_4/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:g
dense_4/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       M
dense_4/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:a
dense_4/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : Û
dense_4/Tensordot/GatherV2GatherV2 dense_4/Tensordot/Shape:output:0dense_4/Tensordot/free:output:0(dense_4/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:c
!dense_4/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ß
dense_4/Tensordot/GatherV2_1GatherV2 dense_4/Tensordot/Shape:output:0dense_4/Tensordot/axes:output:0*dense_4/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense_4/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
dense_4/Tensordot/ProdProd#dense_4/Tensordot/GatherV2:output:0 dense_4/Tensordot/Const:output:0*
T0*
_output_shapes
: c
dense_4/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
dense_4/Tensordot/Prod_1Prod%dense_4/Tensordot/GatherV2_1:output:0"dense_4/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
dense_4/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ¼
dense_4/Tensordot/concatConcatV2dense_4/Tensordot/free:output:0dense_4/Tensordot/axes:output:0&dense_4/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:
dense_4/Tensordot/stackPackdense_4/Tensordot/Prod:output:0!dense_4/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:
dense_4/Tensordot/transpose	Transposeinputs!dense_4/Tensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¢
dense_4/Tensordot/ReshapeReshapedense_4/Tensordot/transpose:y:0 dense_4/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¢
dense_4/Tensordot/MatMulMatMul"dense_4/Tensordot/Reshape:output:0(dense_4/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ c
dense_4/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: a
dense_4/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ç
dense_4/Tensordot/concat_1ConcatV2#dense_4/Tensordot/GatherV2:output:0"dense_4/Tensordot/Const_2:output:0(dense_4/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
dense_4/TensordotReshape"dense_4/Tensordot/MatMul:product:0#dense_4/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
dense_4/BiasAddBiasAdddense_4/Tensordot:output:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ e
dense_4/ReluReludense_4/BiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 dense_5/Tensordot/ReadVariableOpReadVariableOp)dense_5_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype0`
dense_5/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:g
dense_5/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       a
dense_5/Tensordot/ShapeShapedense_4/Relu:activations:0*
T0*
_output_shapes
:a
dense_5/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : Û
dense_5/Tensordot/GatherV2GatherV2 dense_5/Tensordot/Shape:output:0dense_5/Tensordot/free:output:0(dense_5/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:c
!dense_5/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ß
dense_5/Tensordot/GatherV2_1GatherV2 dense_5/Tensordot/Shape:output:0dense_5/Tensordot/axes:output:0*dense_5/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense_5/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
dense_5/Tensordot/ProdProd#dense_5/Tensordot/GatherV2:output:0 dense_5/Tensordot/Const:output:0*
T0*
_output_shapes
: c
dense_5/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
dense_5/Tensordot/Prod_1Prod%dense_5/Tensordot/GatherV2_1:output:0"dense_5/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
dense_5/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ¼
dense_5/Tensordot/concatConcatV2dense_5/Tensordot/free:output:0dense_5/Tensordot/axes:output:0&dense_5/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:
dense_5/Tensordot/stackPackdense_5/Tensordot/Prod:output:0!dense_5/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:
dense_5/Tensordot/transpose	Transposedense_4/Relu:activations:0!dense_5/Tensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¢
dense_5/Tensordot/ReshapeReshapedense_5/Tensordot/transpose:y:0 dense_5/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¢
dense_5/Tensordot/MatMulMatMul"dense_5/Tensordot/Reshape:output:0(dense_5/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ c
dense_5/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: a
dense_5/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ç
dense_5/Tensordot/concat_1ConcatV2#dense_5/Tensordot/GatherV2:output:0"dense_5/Tensordot/Const_2:output:0(dense_5/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
dense_5/TensordotReshape"dense_5/Tensordot/MatMul:product:0#dense_5/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
dense_5/BiasAddBiasAdddense_5/Tensordot:output:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ l
IdentityIdentitydense_5/BiasAdd:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Î
NoOpNoOp^dense_4/BiasAdd/ReadVariableOp!^dense_4/Tensordot/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp!^dense_5/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ : : : : 2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2D
 dense_4/Tensordot/ReadVariableOp dense_4/Tensordot/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2D
 dense_5/Tensordot/ReadVariableOp dense_5/Tensordot/ReadVariableOp:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
§
ø
A__inference_dense_4_layer_call_and_return_conditional_losses_7662

inputs3
!tensordot_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢Tensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:  *
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
value	B : »
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
value	B : ¿
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
:ÿÿÿÿÿÿÿÿÿ 
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ [
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: Y
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
:ÿÿÿÿÿÿÿÿÿ r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ U
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ f
IdentityIdentityRelu:activations:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs


ò
A__inference_dense_6_layer_call_and_return_conditional_losses_7435

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
/
Þ

?__inference_model_layer_call_and_return_conditional_losses_6021
input_1=
*private__token_and_position_embedding_5969:	 =
*private__token_and_position_embedding_5971:	Æ 1
private__transformer_block_5974:  -
private__transformer_block_5976: 1
private__transformer_block_5978:  -
private__transformer_block_5980: 1
private__transformer_block_5982:  -
private__transformer_block_5984: 1
private__transformer_block_5986:  -
private__transformer_block_5988: -
private__transformer_block_5990: -
private__transformer_block_5992: 1
private__transformer_block_5994:  -
private__transformer_block_5996: 1
private__transformer_block_5998:  -
private__transformer_block_6000: -
private__transformer_block_6002: -
private__transformer_block_6004: 
dense_6_6009: 
dense_6_6011:
dense_7_6015:
dense_7_6017:
identity¢dense_6/StatefulPartitionedCall¢dense_7/StatefulPartitionedCall¢!dropout_2/StatefulPartitionedCall¢!dropout_3/StatefulPartitionedCall¢=private__token_and_position_embedding/StatefulPartitionedCall¢2private__transformer_block/StatefulPartitionedCallä
=private__token_and_position_embedding/StatefulPartitionedCallStatefulPartitionedCallinput_1*private__token_and_position_embedding_5969*private__token_and_position_embedding_5971*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *h
fcRa
___inference_private__token_and_position_embedding_layer_call_and_return_conditional_losses_4919á
2private__transformer_block/StatefulPartitionedCallStatefulPartitionedCallFprivate__token_and_position_embedding/StatefulPartitionedCall:output:0private__transformer_block_5974private__transformer_block_5976private__transformer_block_5978private__transformer_block_5980private__transformer_block_5982private__transformer_block_5984private__transformer_block_5986private__transformer_block_5988private__transformer_block_5990private__transformer_block_5992private__transformer_block_5994private__transformer_block_5996private__transformer_block_5998private__transformer_block_6000private__transformer_block_6002private__transformer_block_6004*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_private__transformer_block_layer_call_and_return_conditional_losses_5664
(global_average_pooling1d/PartitionedCallPartitionedCall;private__transformer_block/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_4885ó
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall1global_average_pooling1d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dropout_2_layer_call_and_return_conditional_losses_5363
dense_6/StatefulPartitionedCallStatefulPartitionedCall*dropout_2/StatefulPartitionedCall:output:0dense_6_6009dense_6_6011*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dense_6_layer_call_and_return_conditional_losses_5222
!dropout_3/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0"^dropout_2/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dropout_3_layer_call_and_return_conditional_losses_5330
dense_7/StatefulPartitionedCallStatefulPartitionedCall*dropout_3/StatefulPartitionedCall:output:0dense_7_6015dense_7_6017*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dense_7_layer_call_and_return_conditional_losses_5246w
IdentityIdentity(dense_7/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÇ
NoOpNoOp ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall"^dropout_3/StatefulPartitionedCall>^private__token_and_position_embedding/StatefulPartitionedCall3^private__transformer_block/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : 2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall2F
!dropout_3/StatefulPartitionedCall!dropout_3/StatefulPartitionedCall2~
=private__token_and_position_embedding/StatefulPartitionedCall=private__token_and_position_embedding/StatefulPartitionedCall2h
2private__transformer_block/StatefulPartitionedCall2private__transformer_block/StatefulPartitionedCall:Q M
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
£Ï
 
__inference__wrapped_model_4682
input_1`
Mmodel_private__token_and_position_embedding_embedding_1_embedding_lookup_4411:	 ^
Kmodel_private__token_and_position_embedding_embedding_embedding_lookup_4417:	Æ }
kmodel_private__transformer_block_private__multi_head_self_attention_dense_tensordot_readvariableop_resource:  w
imodel_private__transformer_block_private__multi_head_self_attention_dense_biasadd_readvariableop_resource: 
mmodel_private__transformer_block_private__multi_head_self_attention_dense_1_tensordot_readvariableop_resource:  y
kmodel_private__transformer_block_private__multi_head_self_attention_dense_1_biasadd_readvariableop_resource: 
mmodel_private__transformer_block_private__multi_head_self_attention_dense_2_tensordot_readvariableop_resource:  y
kmodel_private__transformer_block_private__multi_head_self_attention_dense_2_biasadd_readvariableop_resource: 
mmodel_private__transformer_block_private__multi_head_self_attention_dense_3_tensordot_readvariableop_resource:  y
kmodel_private__transformer_block_private__multi_head_self_attention_dense_3_biasadd_readvariableop_resource: h
Zmodel_private__transformer_block_layer_normalization_batchnorm_mul_readvariableop_resource: d
Vmodel_private__transformer_block_layer_normalization_batchnorm_readvariableop_resource: g
Umodel_private__transformer_block_sequential_dense_4_tensordot_readvariableop_resource:  a
Smodel_private__transformer_block_sequential_dense_4_biasadd_readvariableop_resource: g
Umodel_private__transformer_block_sequential_dense_5_tensordot_readvariableop_resource:  a
Smodel_private__transformer_block_sequential_dense_5_biasadd_readvariableop_resource: j
\model_private__transformer_block_layer_normalization_1_batchnorm_mul_readvariableop_resource: f
Xmodel_private__transformer_block_layer_normalization_1_batchnorm_readvariableop_resource: >
,model_dense_6_matmul_readvariableop_resource: ;
-model_dense_6_biasadd_readvariableop_resource:>
,model_dense_7_matmul_readvariableop_resource:;
-model_dense_7_biasadd_readvariableop_resource:
identity¢$model/dense_6/BiasAdd/ReadVariableOp¢#model/dense_6/MatMul/ReadVariableOp¢$model/dense_7/BiasAdd/ReadVariableOp¢#model/dense_7/MatMul/ReadVariableOp¢Fmodel/private__token_and_position_embedding/embedding/embedding_lookup¢Hmodel/private__token_and_position_embedding/embedding_1/embedding_lookup¢Mmodel/private__transformer_block/layer_normalization/batchnorm/ReadVariableOp¢Qmodel/private__transformer_block/layer_normalization/batchnorm/mul/ReadVariableOp¢Omodel/private__transformer_block/layer_normalization_1/batchnorm/ReadVariableOp¢Smodel/private__transformer_block/layer_normalization_1/batchnorm/mul/ReadVariableOp¢`model/private__transformer_block/private__multi_head_self_attention/dense/BiasAdd/ReadVariableOp¢bmodel/private__transformer_block/private__multi_head_self_attention/dense/Tensordot/ReadVariableOp¢bmodel/private__transformer_block/private__multi_head_self_attention/dense_1/BiasAdd/ReadVariableOp¢dmodel/private__transformer_block/private__multi_head_self_attention/dense_1/Tensordot/ReadVariableOp¢bmodel/private__transformer_block/private__multi_head_self_attention/dense_2/BiasAdd/ReadVariableOp¢dmodel/private__transformer_block/private__multi_head_self_attention/dense_2/Tensordot/ReadVariableOp¢bmodel/private__transformer_block/private__multi_head_self_attention/dense_3/BiasAdd/ReadVariableOp¢dmodel/private__transformer_block/private__multi_head_self_attention/dense_3/Tensordot/ReadVariableOp¢Jmodel/private__transformer_block/sequential/dense_4/BiasAdd/ReadVariableOp¢Lmodel/private__transformer_block/sequential/dense_4/Tensordot/ReadVariableOp¢Jmodel/private__transformer_block/sequential/dense_5/BiasAdd/ReadVariableOp¢Lmodel/private__transformer_block/sequential/dense_5/Tensordot/ReadVariableOph
1model/private__token_and_position_embedding/ShapeShapeinput_1*
T0*
_output_shapes
:
?model/private__token_and_position_embedding/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ
Amodel/private__token_and_position_embedding/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
Amodel/private__token_and_position_embedding/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:­
9model/private__token_and_position_embedding/strided_sliceStridedSlice:model/private__token_and_position_embedding/Shape:output:0Hmodel/private__token_and_position_embedding/strided_slice/stack:output:0Jmodel/private__token_and_position_embedding/strided_slice/stack_1:output:0Jmodel/private__token_and_position_embedding/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masky
7model/private__token_and_position_embedding/range/startConst*
_output_shapes
: *
dtype0*
value	B : y
7model/private__token_and_position_embedding/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :
1model/private__token_and_position_embedding/rangeRange@model/private__token_and_position_embedding/range/start:output:0Bmodel/private__token_and_position_embedding/strided_slice:output:0@model/private__token_and_position_embedding/range/delta:output:0*
_output_shapes	
:
Hmodel/private__token_and_position_embedding/embedding_1/embedding_lookupResourceGatherMmodel_private__token_and_position_embedding_embedding_1_embedding_lookup_4411:model/private__token_and_position_embedding/range:output:0*
Tindices0*`
_classV
TRloc:@model/private__token_and_position_embedding/embedding_1/embedding_lookup/4411*
_output_shapes
:	 *
dtype0¼
Qmodel/private__token_and_position_embedding/embedding_1/embedding_lookup/IdentityIdentityQmodel/private__token_and_position_embedding/embedding_1/embedding_lookup:output:0*
T0*`
_classV
TRloc:@model/private__token_and_position_embedding/embedding_1/embedding_lookup/4411*
_output_shapes
:	 å
Smodel/private__token_and_position_embedding/embedding_1/embedding_lookup/Identity_1IdentityZmodel/private__token_and_position_embedding/embedding_1/embedding_lookup/Identity:output:0*
T0*
_output_shapes
:	 
:model/private__token_and_position_embedding/embedding/CastCastinput_1*

DstT0*

SrcT0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Fmodel/private__token_and_position_embedding/embedding/embedding_lookupResourceGatherKmodel_private__token_and_position_embedding_embedding_embedding_lookup_4417>model/private__token_and_position_embedding/embedding/Cast:y:0*
Tindices0*^
_classT
RPloc:@model/private__token_and_position_embedding/embedding/embedding_lookup/4417*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
dtype0Ã
Omodel/private__token_and_position_embedding/embedding/embedding_lookup/IdentityIdentityOmodel/private__token_and_position_embedding/embedding/embedding_lookup:output:0*
T0*^
_classT
RPloc:@model/private__token_and_position_embedding/embedding/embedding_lookup/4417*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ î
Qmodel/private__token_and_position_embedding/embedding/embedding_lookup/Identity_1IdentityXmodel/private__token_and_position_embedding/embedding/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ©
/model/private__token_and_position_embedding/addAddV2Zmodel/private__token_and_position_embedding/embedding/embedding_lookup/Identity_1:output:0\model/private__token_and_position_embedding/embedding_1/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¬
Imodel/private__transformer_block/private__multi_head_self_attention/ShapeShape3model/private__token_and_position_embedding/add:z:0*
T0*
_output_shapes
:¡
Wmodel/private__transformer_block/private__multi_head_self_attention/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: £
Ymodel/private__transformer_block/private__multi_head_self_attention/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:£
Ymodel/private__transformer_block/private__multi_head_self_attention/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¥
Qmodel/private__transformer_block/private__multi_head_self_attention/strided_sliceStridedSliceRmodel/private__transformer_block/private__multi_head_self_attention/Shape:output:0`model/private__transformer_block/private__multi_head_self_attention/strided_slice/stack:output:0bmodel/private__transformer_block/private__multi_head_self_attention/strided_slice/stack_1:output:0bmodel/private__transformer_block/private__multi_head_self_attention/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
bmodel/private__transformer_block/private__multi_head_self_attention/dense/Tensordot/ReadVariableOpReadVariableOpkmodel_private__transformer_block_private__multi_head_self_attention_dense_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype0¢
Xmodel/private__transformer_block/private__multi_head_self_attention/dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:©
Xmodel/private__transformer_block/private__multi_head_self_attention/dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       ¼
Ymodel/private__transformer_block/private__multi_head_self_attention/dense/Tensordot/ShapeShape3model/private__token_and_position_embedding/add:z:0*
T0*
_output_shapes
:£
amodel/private__transformer_block/private__multi_head_self_attention/dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ã
\model/private__transformer_block/private__multi_head_self_attention/dense/Tensordot/GatherV2GatherV2bmodel/private__transformer_block/private__multi_head_self_attention/dense/Tensordot/Shape:output:0amodel/private__transformer_block/private__multi_head_self_attention/dense/Tensordot/free:output:0jmodel/private__transformer_block/private__multi_head_self_attention/dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:¥
cmodel/private__transformer_block/private__multi_head_self_attention/dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ç
^model/private__transformer_block/private__multi_head_self_attention/dense/Tensordot/GatherV2_1GatherV2bmodel/private__transformer_block/private__multi_head_self_attention/dense/Tensordot/Shape:output:0amodel/private__transformer_block/private__multi_head_self_attention/dense/Tensordot/axes:output:0lmodel/private__transformer_block/private__multi_head_self_attention/dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:£
Ymodel/private__transformer_block/private__multi_head_self_attention/dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: Ì
Xmodel/private__transformer_block/private__multi_head_self_attention/dense/Tensordot/ProdProdemodel/private__transformer_block/private__multi_head_self_attention/dense/Tensordot/GatherV2:output:0bmodel/private__transformer_block/private__multi_head_self_attention/dense/Tensordot/Const:output:0*
T0*
_output_shapes
: ¥
[model/private__transformer_block/private__multi_head_self_attention/dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: Ò
Zmodel/private__transformer_block/private__multi_head_self_attention/dense/Tensordot/Prod_1Prodgmodel/private__transformer_block/private__multi_head_self_attention/dense/Tensordot/GatherV2_1:output:0dmodel/private__transformer_block/private__multi_head_self_attention/dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: ¡
_model/private__transformer_block/private__multi_head_self_attention/dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ä
Zmodel/private__transformer_block/private__multi_head_self_attention/dense/Tensordot/concatConcatV2amodel/private__transformer_block/private__multi_head_self_attention/dense/Tensordot/free:output:0amodel/private__transformer_block/private__multi_head_self_attention/dense/Tensordot/axes:output:0hmodel/private__transformer_block/private__multi_head_self_attention/dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:×
Ymodel/private__transformer_block/private__multi_head_self_attention/dense/Tensordot/stackPackamodel/private__transformer_block/private__multi_head_self_attention/dense/Tensordot/Prod:output:0cmodel/private__transformer_block/private__multi_head_self_attention/dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:»
]model/private__transformer_block/private__multi_head_self_attention/dense/Tensordot/transpose	Transpose3model/private__token_and_position_embedding/add:z:0cmodel/private__transformer_block/private__multi_head_self_attention/dense/Tensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ è
[model/private__transformer_block/private__multi_head_self_attention/dense/Tensordot/ReshapeReshapeamodel/private__transformer_block/private__multi_head_self_attention/dense/Tensordot/transpose:y:0bmodel/private__transformer_block/private__multi_head_self_attention/dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿè
Zmodel/private__transformer_block/private__multi_head_self_attention/dense/Tensordot/MatMulMatMuldmodel/private__transformer_block/private__multi_head_self_attention/dense/Tensordot/Reshape:output:0jmodel/private__transformer_block/private__multi_head_self_attention/dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¥
[model/private__transformer_block/private__multi_head_self_attention/dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: £
amodel/private__transformer_block/private__multi_head_self_attention/dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ï
\model/private__transformer_block/private__multi_head_self_attention/dense/Tensordot/concat_1ConcatV2emodel/private__transformer_block/private__multi_head_self_attention/dense/Tensordot/GatherV2:output:0dmodel/private__transformer_block/private__multi_head_self_attention/dense/Tensordot/Const_2:output:0jmodel/private__transformer_block/private__multi_head_self_attention/dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:â
Smodel/private__transformer_block/private__multi_head_self_attention/dense/TensordotReshapedmodel/private__transformer_block/private__multi_head_self_attention/dense/Tensordot/MatMul:product:0emodel/private__transformer_block/private__multi_head_self_attention/dense/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
`model/private__transformer_block/private__multi_head_self_attention/dense/BiasAdd/ReadVariableOpReadVariableOpimodel_private__transformer_block_private__multi_head_self_attention_dense_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Û
Qmodel/private__transformer_block/private__multi_head_self_attention/dense/BiasAddBiasAdd\model/private__transformer_block/private__multi_head_self_attention/dense/Tensordot:output:0hmodel/private__transformer_block/private__multi_head_self_attention/dense/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
dmodel/private__transformer_block/private__multi_head_self_attention/dense_1/Tensordot/ReadVariableOpReadVariableOpmmodel_private__transformer_block_private__multi_head_self_attention_dense_1_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype0¤
Zmodel/private__transformer_block/private__multi_head_self_attention/dense_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:«
Zmodel/private__transformer_block/private__multi_head_self_attention/dense_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       ¾
[model/private__transformer_block/private__multi_head_self_attention/dense_1/Tensordot/ShapeShape3model/private__token_and_position_embedding/add:z:0*
T0*
_output_shapes
:¥
cmodel/private__transformer_block/private__multi_head_self_attention/dense_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ë
^model/private__transformer_block/private__multi_head_self_attention/dense_1/Tensordot/GatherV2GatherV2dmodel/private__transformer_block/private__multi_head_self_attention/dense_1/Tensordot/Shape:output:0cmodel/private__transformer_block/private__multi_head_self_attention/dense_1/Tensordot/free:output:0lmodel/private__transformer_block/private__multi_head_self_attention/dense_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:§
emodel/private__transformer_block/private__multi_head_self_attention/dense_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ï
`model/private__transformer_block/private__multi_head_self_attention/dense_1/Tensordot/GatherV2_1GatherV2dmodel/private__transformer_block/private__multi_head_self_attention/dense_1/Tensordot/Shape:output:0cmodel/private__transformer_block/private__multi_head_self_attention/dense_1/Tensordot/axes:output:0nmodel/private__transformer_block/private__multi_head_self_attention/dense_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:¥
[model/private__transformer_block/private__multi_head_self_attention/dense_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: Ò
Zmodel/private__transformer_block/private__multi_head_self_attention/dense_1/Tensordot/ProdProdgmodel/private__transformer_block/private__multi_head_self_attention/dense_1/Tensordot/GatherV2:output:0dmodel/private__transformer_block/private__multi_head_self_attention/dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: §
]model/private__transformer_block/private__multi_head_self_attention/dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: Ø
\model/private__transformer_block/private__multi_head_self_attention/dense_1/Tensordot/Prod_1Prodimodel/private__transformer_block/private__multi_head_self_attention/dense_1/Tensordot/GatherV2_1:output:0fmodel/private__transformer_block/private__multi_head_self_attention/dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: £
amodel/private__transformer_block/private__multi_head_self_attention/dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ì
\model/private__transformer_block/private__multi_head_self_attention/dense_1/Tensordot/concatConcatV2cmodel/private__transformer_block/private__multi_head_self_attention/dense_1/Tensordot/free:output:0cmodel/private__transformer_block/private__multi_head_self_attention/dense_1/Tensordot/axes:output:0jmodel/private__transformer_block/private__multi_head_self_attention/dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:Ý
[model/private__transformer_block/private__multi_head_self_attention/dense_1/Tensordot/stackPackcmodel/private__transformer_block/private__multi_head_self_attention/dense_1/Tensordot/Prod:output:0emodel/private__transformer_block/private__multi_head_self_attention/dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:¿
_model/private__transformer_block/private__multi_head_self_attention/dense_1/Tensordot/transpose	Transpose3model/private__token_and_position_embedding/add:z:0emodel/private__transformer_block/private__multi_head_self_attention/dense_1/Tensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ î
]model/private__transformer_block/private__multi_head_self_attention/dense_1/Tensordot/ReshapeReshapecmodel/private__transformer_block/private__multi_head_self_attention/dense_1/Tensordot/transpose:y:0dmodel/private__transformer_block/private__multi_head_self_attention/dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿî
\model/private__transformer_block/private__multi_head_self_attention/dense_1/Tensordot/MatMulMatMulfmodel/private__transformer_block/private__multi_head_self_attention/dense_1/Tensordot/Reshape:output:0lmodel/private__transformer_block/private__multi_head_self_attention/dense_1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ §
]model/private__transformer_block/private__multi_head_self_attention/dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: ¥
cmodel/private__transformer_block/private__multi_head_self_attention/dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ×
^model/private__transformer_block/private__multi_head_self_attention/dense_1/Tensordot/concat_1ConcatV2gmodel/private__transformer_block/private__multi_head_self_attention/dense_1/Tensordot/GatherV2:output:0fmodel/private__transformer_block/private__multi_head_self_attention/dense_1/Tensordot/Const_2:output:0lmodel/private__transformer_block/private__multi_head_self_attention/dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:è
Umodel/private__transformer_block/private__multi_head_self_attention/dense_1/TensordotReshapefmodel/private__transformer_block/private__multi_head_self_attention/dense_1/Tensordot/MatMul:product:0gmodel/private__transformer_block/private__multi_head_self_attention/dense_1/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
bmodel/private__transformer_block/private__multi_head_self_attention/dense_1/BiasAdd/ReadVariableOpReadVariableOpkmodel_private__transformer_block_private__multi_head_self_attention_dense_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0á
Smodel/private__transformer_block/private__multi_head_self_attention/dense_1/BiasAddBiasAdd^model/private__transformer_block/private__multi_head_self_attention/dense_1/Tensordot:output:0jmodel/private__transformer_block/private__multi_head_self_attention/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
dmodel/private__transformer_block/private__multi_head_self_attention/dense_2/Tensordot/ReadVariableOpReadVariableOpmmodel_private__transformer_block_private__multi_head_self_attention_dense_2_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype0¤
Zmodel/private__transformer_block/private__multi_head_self_attention/dense_2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:«
Zmodel/private__transformer_block/private__multi_head_self_attention/dense_2/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       ¾
[model/private__transformer_block/private__multi_head_self_attention/dense_2/Tensordot/ShapeShape3model/private__token_and_position_embedding/add:z:0*
T0*
_output_shapes
:¥
cmodel/private__transformer_block/private__multi_head_self_attention/dense_2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ë
^model/private__transformer_block/private__multi_head_self_attention/dense_2/Tensordot/GatherV2GatherV2dmodel/private__transformer_block/private__multi_head_self_attention/dense_2/Tensordot/Shape:output:0cmodel/private__transformer_block/private__multi_head_self_attention/dense_2/Tensordot/free:output:0lmodel/private__transformer_block/private__multi_head_self_attention/dense_2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:§
emodel/private__transformer_block/private__multi_head_self_attention/dense_2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ï
`model/private__transformer_block/private__multi_head_self_attention/dense_2/Tensordot/GatherV2_1GatherV2dmodel/private__transformer_block/private__multi_head_self_attention/dense_2/Tensordot/Shape:output:0cmodel/private__transformer_block/private__multi_head_self_attention/dense_2/Tensordot/axes:output:0nmodel/private__transformer_block/private__multi_head_self_attention/dense_2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:¥
[model/private__transformer_block/private__multi_head_self_attention/dense_2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: Ò
Zmodel/private__transformer_block/private__multi_head_self_attention/dense_2/Tensordot/ProdProdgmodel/private__transformer_block/private__multi_head_self_attention/dense_2/Tensordot/GatherV2:output:0dmodel/private__transformer_block/private__multi_head_self_attention/dense_2/Tensordot/Const:output:0*
T0*
_output_shapes
: §
]model/private__transformer_block/private__multi_head_self_attention/dense_2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: Ø
\model/private__transformer_block/private__multi_head_self_attention/dense_2/Tensordot/Prod_1Prodimodel/private__transformer_block/private__multi_head_self_attention/dense_2/Tensordot/GatherV2_1:output:0fmodel/private__transformer_block/private__multi_head_self_attention/dense_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: £
amodel/private__transformer_block/private__multi_head_self_attention/dense_2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ì
\model/private__transformer_block/private__multi_head_self_attention/dense_2/Tensordot/concatConcatV2cmodel/private__transformer_block/private__multi_head_self_attention/dense_2/Tensordot/free:output:0cmodel/private__transformer_block/private__multi_head_self_attention/dense_2/Tensordot/axes:output:0jmodel/private__transformer_block/private__multi_head_self_attention/dense_2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:Ý
[model/private__transformer_block/private__multi_head_self_attention/dense_2/Tensordot/stackPackcmodel/private__transformer_block/private__multi_head_self_attention/dense_2/Tensordot/Prod:output:0emodel/private__transformer_block/private__multi_head_self_attention/dense_2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:¿
_model/private__transformer_block/private__multi_head_self_attention/dense_2/Tensordot/transpose	Transpose3model/private__token_and_position_embedding/add:z:0emodel/private__transformer_block/private__multi_head_self_attention/dense_2/Tensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ î
]model/private__transformer_block/private__multi_head_self_attention/dense_2/Tensordot/ReshapeReshapecmodel/private__transformer_block/private__multi_head_self_attention/dense_2/Tensordot/transpose:y:0dmodel/private__transformer_block/private__multi_head_self_attention/dense_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿî
\model/private__transformer_block/private__multi_head_self_attention/dense_2/Tensordot/MatMulMatMulfmodel/private__transformer_block/private__multi_head_self_attention/dense_2/Tensordot/Reshape:output:0lmodel/private__transformer_block/private__multi_head_self_attention/dense_2/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ §
]model/private__transformer_block/private__multi_head_self_attention/dense_2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: ¥
cmodel/private__transformer_block/private__multi_head_self_attention/dense_2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ×
^model/private__transformer_block/private__multi_head_self_attention/dense_2/Tensordot/concat_1ConcatV2gmodel/private__transformer_block/private__multi_head_self_attention/dense_2/Tensordot/GatherV2:output:0fmodel/private__transformer_block/private__multi_head_self_attention/dense_2/Tensordot/Const_2:output:0lmodel/private__transformer_block/private__multi_head_self_attention/dense_2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:è
Umodel/private__transformer_block/private__multi_head_self_attention/dense_2/TensordotReshapefmodel/private__transformer_block/private__multi_head_self_attention/dense_2/Tensordot/MatMul:product:0gmodel/private__transformer_block/private__multi_head_self_attention/dense_2/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
bmodel/private__transformer_block/private__multi_head_self_attention/dense_2/BiasAdd/ReadVariableOpReadVariableOpkmodel_private__transformer_block_private__multi_head_self_attention_dense_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0á
Smodel/private__transformer_block/private__multi_head_self_attention/dense_2/BiasAddBiasAdd^model/private__transformer_block/private__multi_head_self_attention/dense_2/Tensordot:output:0jmodel/private__transformer_block/private__multi_head_self_attention/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
Smodel/private__transformer_block/private__multi_head_self_attention/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
Smodel/private__transformer_block/private__multi_head_self_attention/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :
Smodel/private__transformer_block/private__multi_head_self_attention/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :ý
Qmodel/private__transformer_block/private__multi_head_self_attention/Reshape/shapePackZmodel/private__transformer_block/private__multi_head_self_attention/strided_slice:output:0\model/private__transformer_block/private__multi_head_self_attention/Reshape/shape/1:output:0\model/private__transformer_block/private__multi_head_self_attention/Reshape/shape/2:output:0\model/private__transformer_block/private__multi_head_self_attention/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:Ñ
Kmodel/private__transformer_block/private__multi_head_self_attention/ReshapeReshapeZmodel/private__transformer_block/private__multi_head_self_attention/dense/BiasAdd:output:0Zmodel/private__transformer_block/private__multi_head_self_attention/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ«
Rmodel/private__transformer_block/private__multi_head_self_attention/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             Ð
Mmodel/private__transformer_block/private__multi_head_self_attention/transpose	TransposeTmodel/private__transformer_block/private__multi_head_self_attention/Reshape:output:0[model/private__transformer_block/private__multi_head_self_attention/transpose/perm:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
Umodel/private__transformer_block/private__multi_head_self_attention/Reshape_1/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
Umodel/private__transformer_block/private__multi_head_self_attention/Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :
Umodel/private__transformer_block/private__multi_head_self_attention/Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :
Smodel/private__transformer_block/private__multi_head_self_attention/Reshape_1/shapePackZmodel/private__transformer_block/private__multi_head_self_attention/strided_slice:output:0^model/private__transformer_block/private__multi_head_self_attention/Reshape_1/shape/1:output:0^model/private__transformer_block/private__multi_head_self_attention/Reshape_1/shape/2:output:0^model/private__transformer_block/private__multi_head_self_attention/Reshape_1/shape/3:output:0*
N*
T0*
_output_shapes
:×
Mmodel/private__transformer_block/private__multi_head_self_attention/Reshape_1Reshape\model/private__transformer_block/private__multi_head_self_attention/dense_1/BiasAdd:output:0\model/private__transformer_block/private__multi_head_self_attention/Reshape_1/shape:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ­
Tmodel/private__transformer_block/private__multi_head_self_attention/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             Ö
Omodel/private__transformer_block/private__multi_head_self_attention/transpose_1	TransposeVmodel/private__transformer_block/private__multi_head_self_attention/Reshape_1:output:0]model/private__transformer_block/private__multi_head_self_attention/transpose_1/perm:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
Umodel/private__transformer_block/private__multi_head_self_attention/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
Umodel/private__transformer_block/private__multi_head_self_attention/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :
Umodel/private__transformer_block/private__multi_head_self_attention/Reshape_2/shape/3Const*
_output_shapes
: *
dtype0*
value	B :
Smodel/private__transformer_block/private__multi_head_self_attention/Reshape_2/shapePackZmodel/private__transformer_block/private__multi_head_self_attention/strided_slice:output:0^model/private__transformer_block/private__multi_head_self_attention/Reshape_2/shape/1:output:0^model/private__transformer_block/private__multi_head_self_attention/Reshape_2/shape/2:output:0^model/private__transformer_block/private__multi_head_self_attention/Reshape_2/shape/3:output:0*
N*
T0*
_output_shapes
:×
Mmodel/private__transformer_block/private__multi_head_self_attention/Reshape_2Reshape\model/private__transformer_block/private__multi_head_self_attention/dense_2/BiasAdd:output:0\model/private__transformer_block/private__multi_head_self_attention/Reshape_2/shape:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ­
Tmodel/private__transformer_block/private__multi_head_self_attention/transpose_2/permConst*
_output_shapes
:*
dtype0*%
valueB"             Ö
Omodel/private__transformer_block/private__multi_head_self_attention/transpose_2	TransposeVmodel/private__transformer_block/private__multi_head_self_attention/Reshape_2:output:0]model/private__transformer_block/private__multi_head_self_attention/transpose_2/perm:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÜ
Jmodel/private__transformer_block/private__multi_head_self_attention/MatMulBatchMatMulV2Qmodel/private__transformer_block/private__multi_head_self_attention/transpose:y:0Smodel/private__transformer_block/private__multi_head_self_attention/transpose_1:y:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
adj_y(Î
Kmodel/private__transformer_block/private__multi_head_self_attention/Shape_1ShapeSmodel/private__transformer_block/private__multi_head_self_attention/transpose_1:y:0*
T0*
_output_shapes
:¬
Ymodel/private__transformer_block/private__multi_head_self_attention/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ¥
[model/private__transformer_block/private__multi_head_self_attention/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: ¥
[model/private__transformer_block/private__multi_head_self_attention/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¯
Smodel/private__transformer_block/private__multi_head_self_attention/strided_slice_1StridedSliceTmodel/private__transformer_block/private__multi_head_self_attention/Shape_1:output:0bmodel/private__transformer_block/private__multi_head_self_attention/strided_slice_1/stack:output:0dmodel/private__transformer_block/private__multi_head_self_attention/strided_slice_1/stack_1:output:0dmodel/private__transformer_block/private__multi_head_self_attention/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskÞ
Hmodel/private__transformer_block/private__multi_head_self_attention/CastCast\model/private__transformer_block/private__multi_head_self_attention/strided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: ¿
Hmodel/private__transformer_block/private__multi_head_self_attention/SqrtSqrtLmodel/private__transformer_block/private__multi_head_self_attention/Cast:y:0*
T0*
_output_shapes
: Å
Kmodel/private__transformer_block/private__multi_head_self_attention/truedivRealDivSmodel/private__transformer_block/private__multi_head_self_attention/MatMul:output:0Lmodel/private__transformer_block/private__multi_head_self_attention/Sqrt:y:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿó
Kmodel/private__transformer_block/private__multi_head_self_attention/SoftmaxSoftmaxOmodel/private__transformer_block/private__multi_head_self_attention/truediv:z:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÌ
Lmodel/private__transformer_block/private__multi_head_self_attention/MatMul_1BatchMatMulV2Umodel/private__transformer_block/private__multi_head_self_attention/Softmax:softmax:0Smodel/private__transformer_block/private__multi_head_self_attention/transpose_2:y:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ­
Tmodel/private__transformer_block/private__multi_head_self_attention/transpose_3/permConst*
_output_shapes
:*
dtype0*%
valueB"             Õ
Omodel/private__transformer_block/private__multi_head_self_attention/transpose_3	TransposeUmodel/private__transformer_block/private__multi_head_self_attention/MatMul_1:output:0]model/private__transformer_block/private__multi_head_self_attention/transpose_3/perm:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
Umodel/private__transformer_block/private__multi_head_self_attention/Reshape_3/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
Umodel/private__transformer_block/private__multi_head_self_attention/Reshape_3/shape/2Const*
_output_shapes
: *
dtype0*
value	B : ¥
Smodel/private__transformer_block/private__multi_head_self_attention/Reshape_3/shapePackZmodel/private__transformer_block/private__multi_head_self_attention/strided_slice:output:0^model/private__transformer_block/private__multi_head_self_attention/Reshape_3/shape/1:output:0^model/private__transformer_block/private__multi_head_self_attention/Reshape_3/shape/2:output:0*
N*
T0*
_output_shapes
:Ê
Mmodel/private__transformer_block/private__multi_head_self_attention/Reshape_3ReshapeSmodel/private__transformer_block/private__multi_head_self_attention/transpose_3:y:0\model/private__transformer_block/private__multi_head_self_attention/Reshape_3/shape:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
dmodel/private__transformer_block/private__multi_head_self_attention/dense_3/Tensordot/ReadVariableOpReadVariableOpmmodel_private__transformer_block_private__multi_head_self_attention_dense_3_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype0¤
Zmodel/private__transformer_block/private__multi_head_self_attention/dense_3/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:«
Zmodel/private__transformer_block/private__multi_head_self_attention/dense_3/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       á
[model/private__transformer_block/private__multi_head_self_attention/dense_3/Tensordot/ShapeShapeVmodel/private__transformer_block/private__multi_head_self_attention/Reshape_3:output:0*
T0*
_output_shapes
:¥
cmodel/private__transformer_block/private__multi_head_self_attention/dense_3/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ë
^model/private__transformer_block/private__multi_head_self_attention/dense_3/Tensordot/GatherV2GatherV2dmodel/private__transformer_block/private__multi_head_self_attention/dense_3/Tensordot/Shape:output:0cmodel/private__transformer_block/private__multi_head_self_attention/dense_3/Tensordot/free:output:0lmodel/private__transformer_block/private__multi_head_self_attention/dense_3/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:§
emodel/private__transformer_block/private__multi_head_self_attention/dense_3/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ï
`model/private__transformer_block/private__multi_head_self_attention/dense_3/Tensordot/GatherV2_1GatherV2dmodel/private__transformer_block/private__multi_head_self_attention/dense_3/Tensordot/Shape:output:0cmodel/private__transformer_block/private__multi_head_self_attention/dense_3/Tensordot/axes:output:0nmodel/private__transformer_block/private__multi_head_self_attention/dense_3/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:¥
[model/private__transformer_block/private__multi_head_self_attention/dense_3/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: Ò
Zmodel/private__transformer_block/private__multi_head_self_attention/dense_3/Tensordot/ProdProdgmodel/private__transformer_block/private__multi_head_self_attention/dense_3/Tensordot/GatherV2:output:0dmodel/private__transformer_block/private__multi_head_self_attention/dense_3/Tensordot/Const:output:0*
T0*
_output_shapes
: §
]model/private__transformer_block/private__multi_head_self_attention/dense_3/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: Ø
\model/private__transformer_block/private__multi_head_self_attention/dense_3/Tensordot/Prod_1Prodimodel/private__transformer_block/private__multi_head_self_attention/dense_3/Tensordot/GatherV2_1:output:0fmodel/private__transformer_block/private__multi_head_self_attention/dense_3/Tensordot/Const_1:output:0*
T0*
_output_shapes
: £
amodel/private__transformer_block/private__multi_head_self_attention/dense_3/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ì
\model/private__transformer_block/private__multi_head_self_attention/dense_3/Tensordot/concatConcatV2cmodel/private__transformer_block/private__multi_head_self_attention/dense_3/Tensordot/free:output:0cmodel/private__transformer_block/private__multi_head_self_attention/dense_3/Tensordot/axes:output:0jmodel/private__transformer_block/private__multi_head_self_attention/dense_3/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:Ý
[model/private__transformer_block/private__multi_head_self_attention/dense_3/Tensordot/stackPackcmodel/private__transformer_block/private__multi_head_self_attention/dense_3/Tensordot/Prod:output:0emodel/private__transformer_block/private__multi_head_self_attention/dense_3/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:ê
_model/private__transformer_block/private__multi_head_self_attention/dense_3/Tensordot/transpose	TransposeVmodel/private__transformer_block/private__multi_head_self_attention/Reshape_3:output:0emodel/private__transformer_block/private__multi_head_self_attention/dense_3/Tensordot/concat:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ î
]model/private__transformer_block/private__multi_head_self_attention/dense_3/Tensordot/ReshapeReshapecmodel/private__transformer_block/private__multi_head_self_attention/dense_3/Tensordot/transpose:y:0dmodel/private__transformer_block/private__multi_head_self_attention/dense_3/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿî
\model/private__transformer_block/private__multi_head_self_attention/dense_3/Tensordot/MatMulMatMulfmodel/private__transformer_block/private__multi_head_self_attention/dense_3/Tensordot/Reshape:output:0lmodel/private__transformer_block/private__multi_head_self_attention/dense_3/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ §
]model/private__transformer_block/private__multi_head_self_attention/dense_3/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: ¥
cmodel/private__transformer_block/private__multi_head_self_attention/dense_3/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ×
^model/private__transformer_block/private__multi_head_self_attention/dense_3/Tensordot/concat_1ConcatV2gmodel/private__transformer_block/private__multi_head_self_attention/dense_3/Tensordot/GatherV2:output:0fmodel/private__transformer_block/private__multi_head_self_attention/dense_3/Tensordot/Const_2:output:0lmodel/private__transformer_block/private__multi_head_self_attention/dense_3/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:ð
Umodel/private__transformer_block/private__multi_head_self_attention/dense_3/TensordotReshapefmodel/private__transformer_block/private__multi_head_self_attention/dense_3/Tensordot/MatMul:product:0gmodel/private__transformer_block/private__multi_head_self_attention/dense_3/Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
bmodel/private__transformer_block/private__multi_head_self_attention/dense_3/BiasAdd/ReadVariableOpReadVariableOpkmodel_private__transformer_block_private__multi_head_self_attention_dense_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0é
Smodel/private__transformer_block/private__multi_head_self_attention/dense_3/BiasAddBiasAdd^model/private__transformer_block/private__multi_head_self_attention/dense_3/Tensordot:output:0jmodel/private__transformer_block/private__multi_head_self_attention/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ Ú
1model/private__transformer_block/dropout/IdentityIdentity\model/private__transformer_block/private__multi_head_self_attention/dense_3/BiasAdd:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ Õ
$model/private__transformer_block/addAddV23model/private__token_and_position_embedding/add:z:0:model/private__transformer_block/dropout/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
Smodel/private__transformer_block/layer_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
Amodel/private__transformer_block/layer_normalization/moments/meanMean(model/private__transformer_block/add:z:0\model/private__transformer_block/layer_normalization/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(Ü
Imodel/private__transformer_block/layer_normalization/moments/StopGradientStopGradientJmodel/private__transformer_block/layer_normalization/moments/mean:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Nmodel/private__transformer_block/layer_normalization/moments/SquaredDifferenceSquaredDifference(model/private__transformer_block/add:z:0Rmodel/private__transformer_block/layer_normalization/moments/StopGradient:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¡
Wmodel/private__transformer_block/layer_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:Ë
Emodel/private__transformer_block/layer_normalization/moments/varianceMeanRmodel/private__transformer_block/layer_normalization/moments/SquaredDifference:z:0`model/private__transformer_block/layer_normalization/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(
Dmodel/private__transformer_block/layer_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½75¡
Bmodel/private__transformer_block/layer_normalization/batchnorm/addAddV2Nmodel/private__transformer_block/layer_normalization/moments/variance:output:0Mmodel/private__transformer_block/layer_normalization/batchnorm/add/y:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÌ
Dmodel/private__transformer_block/layer_normalization/batchnorm/RsqrtRsqrtFmodel/private__transformer_block/layer_normalization/batchnorm/add:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
Qmodel/private__transformer_block/layer_normalization/batchnorm/mul/ReadVariableOpReadVariableOpZmodel_private__transformer_block_layer_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype0¥
Bmodel/private__transformer_block/layer_normalization/batchnorm/mulMulHmodel/private__transformer_block/layer_normalization/batchnorm/Rsqrt:y:0Ymodel/private__transformer_block/layer_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ô
Dmodel/private__transformer_block/layer_normalization/batchnorm/mul_1Mul(model/private__transformer_block/add:z:0Fmodel/private__transformer_block/layer_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
Dmodel/private__transformer_block/layer_normalization/batchnorm/mul_2MulJmodel/private__transformer_block/layer_normalization/moments/mean:output:0Fmodel/private__transformer_block/layer_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ à
Mmodel/private__transformer_block/layer_normalization/batchnorm/ReadVariableOpReadVariableOpVmodel_private__transformer_block_layer_normalization_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype0¡
Bmodel/private__transformer_block/layer_normalization/batchnorm/subSubUmodel/private__transformer_block/layer_normalization/batchnorm/ReadVariableOp:value:0Hmodel/private__transformer_block/layer_normalization/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
Dmodel/private__transformer_block/layer_normalization/batchnorm/add_1AddV2Hmodel/private__transformer_block/layer_normalization/batchnorm/mul_1:z:0Fmodel/private__transformer_block/layer_normalization/batchnorm/sub:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ â
Lmodel/private__transformer_block/sequential/dense_4/Tensordot/ReadVariableOpReadVariableOpUmodel_private__transformer_block_sequential_dense_4_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype0
Bmodel/private__transformer_block/sequential/dense_4/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:
Bmodel/private__transformer_block/sequential/dense_4/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       »
Cmodel/private__transformer_block/sequential/dense_4/Tensordot/ShapeShapeHmodel/private__transformer_block/layer_normalization/batchnorm/add_1:z:0*
T0*
_output_shapes
:
Kmodel/private__transformer_block/sequential/dense_4/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Fmodel/private__transformer_block/sequential/dense_4/Tensordot/GatherV2GatherV2Lmodel/private__transformer_block/sequential/dense_4/Tensordot/Shape:output:0Kmodel/private__transformer_block/sequential/dense_4/Tensordot/free:output:0Tmodel/private__transformer_block/sequential/dense_4/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:
Mmodel/private__transformer_block/sequential/dense_4/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Hmodel/private__transformer_block/sequential/dense_4/Tensordot/GatherV2_1GatherV2Lmodel/private__transformer_block/sequential/dense_4/Tensordot/Shape:output:0Kmodel/private__transformer_block/sequential/dense_4/Tensordot/axes:output:0Vmodel/private__transformer_block/sequential/dense_4/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:
Cmodel/private__transformer_block/sequential/dense_4/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
Bmodel/private__transformer_block/sequential/dense_4/Tensordot/ProdProdOmodel/private__transformer_block/sequential/dense_4/Tensordot/GatherV2:output:0Lmodel/private__transformer_block/sequential/dense_4/Tensordot/Const:output:0*
T0*
_output_shapes
: 
Emodel/private__transformer_block/sequential/dense_4/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
Dmodel/private__transformer_block/sequential/dense_4/Tensordot/Prod_1ProdQmodel/private__transformer_block/sequential/dense_4/Tensordot/GatherV2_1:output:0Nmodel/private__transformer_block/sequential/dense_4/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 
Imodel/private__transformer_block/sequential/dense_4/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ì
Dmodel/private__transformer_block/sequential/dense_4/Tensordot/concatConcatV2Kmodel/private__transformer_block/sequential/dense_4/Tensordot/free:output:0Kmodel/private__transformer_block/sequential/dense_4/Tensordot/axes:output:0Rmodel/private__transformer_block/sequential/dense_4/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:
Cmodel/private__transformer_block/sequential/dense_4/Tensordot/stackPackKmodel/private__transformer_block/sequential/dense_4/Tensordot/Prod:output:0Mmodel/private__transformer_block/sequential/dense_4/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:¤
Gmodel/private__transformer_block/sequential/dense_4/Tensordot/transpose	TransposeHmodel/private__transformer_block/layer_normalization/batchnorm/add_1:z:0Mmodel/private__transformer_block/sequential/dense_4/Tensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¦
Emodel/private__transformer_block/sequential/dense_4/Tensordot/ReshapeReshapeKmodel/private__transformer_block/sequential/dense_4/Tensordot/transpose:y:0Lmodel/private__transformer_block/sequential/dense_4/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¦
Dmodel/private__transformer_block/sequential/dense_4/Tensordot/MatMulMatMulNmodel/private__transformer_block/sequential/dense_4/Tensordot/Reshape:output:0Tmodel/private__transformer_block/sequential/dense_4/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
Emodel/private__transformer_block/sequential/dense_4/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 
Kmodel/private__transformer_block/sequential/dense_4/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ÷
Fmodel/private__transformer_block/sequential/dense_4/Tensordot/concat_1ConcatV2Omodel/private__transformer_block/sequential/dense_4/Tensordot/GatherV2:output:0Nmodel/private__transformer_block/sequential/dense_4/Tensordot/Const_2:output:0Tmodel/private__transformer_block/sequential/dense_4/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
: 
=model/private__transformer_block/sequential/dense_4/TensordotReshapeNmodel/private__transformer_block/sequential/dense_4/Tensordot/MatMul:product:0Omodel/private__transformer_block/sequential/dense_4/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ú
Jmodel/private__transformer_block/sequential/dense_4/BiasAdd/ReadVariableOpReadVariableOpSmodel_private__transformer_block_sequential_dense_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
;model/private__transformer_block/sequential/dense_4/BiasAddBiasAddFmodel/private__transformer_block/sequential/dense_4/Tensordot:output:0Rmodel/private__transformer_block/sequential/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ½
8model/private__transformer_block/sequential/dense_4/ReluReluDmodel/private__transformer_block/sequential/dense_4/BiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ â
Lmodel/private__transformer_block/sequential/dense_5/Tensordot/ReadVariableOpReadVariableOpUmodel_private__transformer_block_sequential_dense_5_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype0
Bmodel/private__transformer_block/sequential/dense_5/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:
Bmodel/private__transformer_block/sequential/dense_5/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       ¹
Cmodel/private__transformer_block/sequential/dense_5/Tensordot/ShapeShapeFmodel/private__transformer_block/sequential/dense_4/Relu:activations:0*
T0*
_output_shapes
:
Kmodel/private__transformer_block/sequential/dense_5/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Fmodel/private__transformer_block/sequential/dense_5/Tensordot/GatherV2GatherV2Lmodel/private__transformer_block/sequential/dense_5/Tensordot/Shape:output:0Kmodel/private__transformer_block/sequential/dense_5/Tensordot/free:output:0Tmodel/private__transformer_block/sequential/dense_5/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:
Mmodel/private__transformer_block/sequential/dense_5/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Hmodel/private__transformer_block/sequential/dense_5/Tensordot/GatherV2_1GatherV2Lmodel/private__transformer_block/sequential/dense_5/Tensordot/Shape:output:0Kmodel/private__transformer_block/sequential/dense_5/Tensordot/axes:output:0Vmodel/private__transformer_block/sequential/dense_5/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:
Cmodel/private__transformer_block/sequential/dense_5/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
Bmodel/private__transformer_block/sequential/dense_5/Tensordot/ProdProdOmodel/private__transformer_block/sequential/dense_5/Tensordot/GatherV2:output:0Lmodel/private__transformer_block/sequential/dense_5/Tensordot/Const:output:0*
T0*
_output_shapes
: 
Emodel/private__transformer_block/sequential/dense_5/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
Dmodel/private__transformer_block/sequential/dense_5/Tensordot/Prod_1ProdQmodel/private__transformer_block/sequential/dense_5/Tensordot/GatherV2_1:output:0Nmodel/private__transformer_block/sequential/dense_5/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 
Imodel/private__transformer_block/sequential/dense_5/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ì
Dmodel/private__transformer_block/sequential/dense_5/Tensordot/concatConcatV2Kmodel/private__transformer_block/sequential/dense_5/Tensordot/free:output:0Kmodel/private__transformer_block/sequential/dense_5/Tensordot/axes:output:0Rmodel/private__transformer_block/sequential/dense_5/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:
Cmodel/private__transformer_block/sequential/dense_5/Tensordot/stackPackKmodel/private__transformer_block/sequential/dense_5/Tensordot/Prod:output:0Mmodel/private__transformer_block/sequential/dense_5/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:¢
Gmodel/private__transformer_block/sequential/dense_5/Tensordot/transpose	TransposeFmodel/private__transformer_block/sequential/dense_4/Relu:activations:0Mmodel/private__transformer_block/sequential/dense_5/Tensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¦
Emodel/private__transformer_block/sequential/dense_5/Tensordot/ReshapeReshapeKmodel/private__transformer_block/sequential/dense_5/Tensordot/transpose:y:0Lmodel/private__transformer_block/sequential/dense_5/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¦
Dmodel/private__transformer_block/sequential/dense_5/Tensordot/MatMulMatMulNmodel/private__transformer_block/sequential/dense_5/Tensordot/Reshape:output:0Tmodel/private__transformer_block/sequential/dense_5/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
Emodel/private__transformer_block/sequential/dense_5/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 
Kmodel/private__transformer_block/sequential/dense_5/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ÷
Fmodel/private__transformer_block/sequential/dense_5/Tensordot/concat_1ConcatV2Omodel/private__transformer_block/sequential/dense_5/Tensordot/GatherV2:output:0Nmodel/private__transformer_block/sequential/dense_5/Tensordot/Const_2:output:0Tmodel/private__transformer_block/sequential/dense_5/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
: 
=model/private__transformer_block/sequential/dense_5/TensordotReshapeNmodel/private__transformer_block/sequential/dense_5/Tensordot/MatMul:product:0Omodel/private__transformer_block/sequential/dense_5/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ú
Jmodel/private__transformer_block/sequential/dense_5/BiasAdd/ReadVariableOpReadVariableOpSmodel_private__transformer_block_sequential_dense_5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
;model/private__transformer_block/sequential/dense_5/BiasAddBiasAddFmodel/private__transformer_block/sequential/dense_5/Tensordot:output:0Rmodel/private__transformer_block/sequential/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¼
3model/private__transformer_block/dropout_1/IdentityIdentityDmodel/private__transformer_block/sequential/dense_5/BiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ î
&model/private__transformer_block/add_1AddV2Hmodel/private__transformer_block/layer_normalization/batchnorm/add_1:z:0<model/private__transformer_block/dropout_1/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
Umodel/private__transformer_block/layer_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
Cmodel/private__transformer_block/layer_normalization_1/moments/meanMean*model/private__transformer_block/add_1:z:0^model/private__transformer_block/layer_normalization_1/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(à
Kmodel/private__transformer_block/layer_normalization_1/moments/StopGradientStopGradientLmodel/private__transformer_block/layer_normalization_1/moments/mean:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Pmodel/private__transformer_block/layer_normalization_1/moments/SquaredDifferenceSquaredDifference*model/private__transformer_block/add_1:z:0Tmodel/private__transformer_block/layer_normalization_1/moments/StopGradient:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ £
Ymodel/private__transformer_block/layer_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:Ñ
Gmodel/private__transformer_block/layer_normalization_1/moments/varianceMeanTmodel/private__transformer_block/layer_normalization_1/moments/SquaredDifference:z:0bmodel/private__transformer_block/layer_normalization_1/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(
Fmodel/private__transformer_block/layer_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½75§
Dmodel/private__transformer_block/layer_normalization_1/batchnorm/addAddV2Pmodel/private__transformer_block/layer_normalization_1/moments/variance:output:0Omodel/private__transformer_block/layer_normalization_1/batchnorm/add/y:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ
Fmodel/private__transformer_block/layer_normalization_1/batchnorm/RsqrtRsqrtHmodel/private__transformer_block/layer_normalization_1/batchnorm/add:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿì
Smodel/private__transformer_block/layer_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOp\model_private__transformer_block_layer_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype0«
Dmodel/private__transformer_block/layer_normalization_1/batchnorm/mulMulJmodel/private__transformer_block/layer_normalization_1/batchnorm/Rsqrt:y:0[model/private__transformer_block/layer_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ú
Fmodel/private__transformer_block/layer_normalization_1/batchnorm/mul_1Mul*model/private__transformer_block/add_1:z:0Hmodel/private__transformer_block/layer_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
Fmodel/private__transformer_block/layer_normalization_1/batchnorm/mul_2MulLmodel/private__transformer_block/layer_normalization_1/moments/mean:output:0Hmodel/private__transformer_block/layer_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ä
Omodel/private__transformer_block/layer_normalization_1/batchnorm/ReadVariableOpReadVariableOpXmodel_private__transformer_block_layer_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype0§
Dmodel/private__transformer_block/layer_normalization_1/batchnorm/subSubWmodel/private__transformer_block/layer_normalization_1/batchnorm/ReadVariableOp:value:0Jmodel/private__transformer_block/layer_normalization_1/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
Fmodel/private__transformer_block/layer_normalization_1/batchnorm/add_1AddV2Jmodel/private__transformer_block/layer_normalization_1/batchnorm/mul_1:z:0Hmodel/private__transformer_block/layer_normalization_1/batchnorm/sub:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ w
5model/global_average_pooling1d/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :é
#model/global_average_pooling1d/MeanMeanJmodel/private__transformer_block/layer_normalization_1/batchnorm/add_1:z:0>model/global_average_pooling1d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
model/dropout_2/IdentityIdentity,model/global_average_pooling1d/Mean:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
#model/dense_6/MatMul/ReadVariableOpReadVariableOp,model_dense_6_matmul_readvariableop_resource*
_output_shapes

: *
dtype0 
model/dense_6/MatMulMatMul!model/dropout_2/Identity:output:0+model/dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$model/dense_6/BiasAdd/ReadVariableOpReadVariableOp-model_dense_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0 
model/dense_6/BiasAddBiasAddmodel/dense_6/MatMul:product:0,model/dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
model/dense_6/ReluRelumodel/dense_6/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
model/dropout_3/IdentityIdentity model/dense_6/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#model/dense_7/MatMul/ReadVariableOpReadVariableOp,model_dense_7_matmul_readvariableop_resource*
_output_shapes

:*
dtype0 
model/dense_7/MatMulMatMul!model/dropout_3/Identity:output:0+model/dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$model/dense_7/BiasAdd/ReadVariableOpReadVariableOp-model_dense_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0 
model/dense_7/BiasAddBiasAddmodel/dense_7/MatMul:product:0,model/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
model/dense_7/SoftmaxSoftmaxmodel/dense_7/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
IdentityIdentitymodel/dense_7/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
NoOpNoOp%^model/dense_6/BiasAdd/ReadVariableOp$^model/dense_6/MatMul/ReadVariableOp%^model/dense_7/BiasAdd/ReadVariableOp$^model/dense_7/MatMul/ReadVariableOpG^model/private__token_and_position_embedding/embedding/embedding_lookupI^model/private__token_and_position_embedding/embedding_1/embedding_lookupN^model/private__transformer_block/layer_normalization/batchnorm/ReadVariableOpR^model/private__transformer_block/layer_normalization/batchnorm/mul/ReadVariableOpP^model/private__transformer_block/layer_normalization_1/batchnorm/ReadVariableOpT^model/private__transformer_block/layer_normalization_1/batchnorm/mul/ReadVariableOpa^model/private__transformer_block/private__multi_head_self_attention/dense/BiasAdd/ReadVariableOpc^model/private__transformer_block/private__multi_head_self_attention/dense/Tensordot/ReadVariableOpc^model/private__transformer_block/private__multi_head_self_attention/dense_1/BiasAdd/ReadVariableOpe^model/private__transformer_block/private__multi_head_self_attention/dense_1/Tensordot/ReadVariableOpc^model/private__transformer_block/private__multi_head_self_attention/dense_2/BiasAdd/ReadVariableOpe^model/private__transformer_block/private__multi_head_self_attention/dense_2/Tensordot/ReadVariableOpc^model/private__transformer_block/private__multi_head_self_attention/dense_3/BiasAdd/ReadVariableOpe^model/private__transformer_block/private__multi_head_self_attention/dense_3/Tensordot/ReadVariableOpK^model/private__transformer_block/sequential/dense_4/BiasAdd/ReadVariableOpM^model/private__transformer_block/sequential/dense_4/Tensordot/ReadVariableOpK^model/private__transformer_block/sequential/dense_5/BiasAdd/ReadVariableOpM^model/private__transformer_block/sequential/dense_5/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : 2L
$model/dense_6/BiasAdd/ReadVariableOp$model/dense_6/BiasAdd/ReadVariableOp2J
#model/dense_6/MatMul/ReadVariableOp#model/dense_6/MatMul/ReadVariableOp2L
$model/dense_7/BiasAdd/ReadVariableOp$model/dense_7/BiasAdd/ReadVariableOp2J
#model/dense_7/MatMul/ReadVariableOp#model/dense_7/MatMul/ReadVariableOp2
Fmodel/private__token_and_position_embedding/embedding/embedding_lookupFmodel/private__token_and_position_embedding/embedding/embedding_lookup2
Hmodel/private__token_and_position_embedding/embedding_1/embedding_lookupHmodel/private__token_and_position_embedding/embedding_1/embedding_lookup2
Mmodel/private__transformer_block/layer_normalization/batchnorm/ReadVariableOpMmodel/private__transformer_block/layer_normalization/batchnorm/ReadVariableOp2¦
Qmodel/private__transformer_block/layer_normalization/batchnorm/mul/ReadVariableOpQmodel/private__transformer_block/layer_normalization/batchnorm/mul/ReadVariableOp2¢
Omodel/private__transformer_block/layer_normalization_1/batchnorm/ReadVariableOpOmodel/private__transformer_block/layer_normalization_1/batchnorm/ReadVariableOp2ª
Smodel/private__transformer_block/layer_normalization_1/batchnorm/mul/ReadVariableOpSmodel/private__transformer_block/layer_normalization_1/batchnorm/mul/ReadVariableOp2Ä
`model/private__transformer_block/private__multi_head_self_attention/dense/BiasAdd/ReadVariableOp`model/private__transformer_block/private__multi_head_self_attention/dense/BiasAdd/ReadVariableOp2È
bmodel/private__transformer_block/private__multi_head_self_attention/dense/Tensordot/ReadVariableOpbmodel/private__transformer_block/private__multi_head_self_attention/dense/Tensordot/ReadVariableOp2È
bmodel/private__transformer_block/private__multi_head_self_attention/dense_1/BiasAdd/ReadVariableOpbmodel/private__transformer_block/private__multi_head_self_attention/dense_1/BiasAdd/ReadVariableOp2Ì
dmodel/private__transformer_block/private__multi_head_self_attention/dense_1/Tensordot/ReadVariableOpdmodel/private__transformer_block/private__multi_head_self_attention/dense_1/Tensordot/ReadVariableOp2È
bmodel/private__transformer_block/private__multi_head_self_attention/dense_2/BiasAdd/ReadVariableOpbmodel/private__transformer_block/private__multi_head_self_attention/dense_2/BiasAdd/ReadVariableOp2Ì
dmodel/private__transformer_block/private__multi_head_self_attention/dense_2/Tensordot/ReadVariableOpdmodel/private__transformer_block/private__multi_head_self_attention/dense_2/Tensordot/ReadVariableOp2È
bmodel/private__transformer_block/private__multi_head_self_attention/dense_3/BiasAdd/ReadVariableOpbmodel/private__transformer_block/private__multi_head_self_attention/dense_3/BiasAdd/ReadVariableOp2Ì
dmodel/private__transformer_block/private__multi_head_self_attention/dense_3/Tensordot/ReadVariableOpdmodel/private__transformer_block/private__multi_head_self_attention/dense_3/Tensordot/ReadVariableOp2
Jmodel/private__transformer_block/sequential/dense_4/BiasAdd/ReadVariableOpJmodel/private__transformer_block/sequential/dense_4/BiasAdd/ReadVariableOp2
Lmodel/private__transformer_block/sequential/dense_4/Tensordot/ReadVariableOpLmodel/private__transformer_block/sequential/dense_4/Tensordot/ReadVariableOp2
Jmodel/private__transformer_block/sequential/dense_5/BiasAdd/ReadVariableOpJmodel/private__transformer_block/sequential/dense_5/BiasAdd/ReadVariableOp2
Lmodel/private__transformer_block/sequential/dense_5/Tensordot/ReadVariableOpLmodel/private__transformer_block/sequential/dense_5/Tensordot/ReadVariableOp:Q M
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
,


?__inference_model_layer_call_and_return_conditional_losses_5253

inputs=
*private__token_and_position_embedding_4920:	 =
*private__token_and_position_embedding_4922:	Æ 1
private__transformer_block_5170:  -
private__transformer_block_5172: 1
private__transformer_block_5174:  -
private__transformer_block_5176: 1
private__transformer_block_5178:  -
private__transformer_block_5180: 1
private__transformer_block_5182:  -
private__transformer_block_5184: -
private__transformer_block_5186: -
private__transformer_block_5188: 1
private__transformer_block_5190:  -
private__transformer_block_5192: 1
private__transformer_block_5194:  -
private__transformer_block_5196: -
private__transformer_block_5198: -
private__transformer_block_5200: 
dense_6_5223: 
dense_6_5225:
dense_7_5247:
dense_7_5249:
identity¢dense_6/StatefulPartitionedCall¢dense_7/StatefulPartitionedCall¢=private__token_and_position_embedding/StatefulPartitionedCall¢2private__transformer_block/StatefulPartitionedCallã
=private__token_and_position_embedding/StatefulPartitionedCallStatefulPartitionedCallinputs*private__token_and_position_embedding_4920*private__token_and_position_embedding_4922*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *h
fcRa
___inference_private__token_and_position_embedding_layer_call_and_return_conditional_losses_4919á
2private__transformer_block/StatefulPartitionedCallStatefulPartitionedCallFprivate__token_and_position_embedding/StatefulPartitionedCall:output:0private__transformer_block_5170private__transformer_block_5172private__transformer_block_5174private__transformer_block_5176private__transformer_block_5178private__transformer_block_5180private__transformer_block_5182private__transformer_block_5184private__transformer_block_5186private__transformer_block_5188private__transformer_block_5190private__transformer_block_5192private__transformer_block_5194private__transformer_block_5196private__transformer_block_5198private__transformer_block_5200*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_private__transformer_block_layer_call_and_return_conditional_losses_5169
(global_average_pooling1d/PartitionedCallPartitionedCall;private__transformer_block/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_4885ã
dropout_2/PartitionedCallPartitionedCall1global_average_pooling1d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dropout_2_layer_call_and_return_conditional_losses_5209
dense_6/StatefulPartitionedCallStatefulPartitionedCall"dropout_2/PartitionedCall:output:0dense_6_5223dense_6_5225*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dense_6_layer_call_and_return_conditional_losses_5222Ú
dropout_3/PartitionedCallPartitionedCall(dense_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dropout_3_layer_call_and_return_conditional_losses_5233
dense_7/StatefulPartitionedCallStatefulPartitionedCall"dropout_3/PartitionedCall:output:0dense_7_5247dense_7_5249*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dense_7_layer_call_and_return_conditional_losses_5246w
IdentityIdentity(dense_7/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿ
NoOpNoOp ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall>^private__token_and_position_embedding/StatefulPartitionedCall3^private__transformer_block/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : 2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2~
=private__token_and_position_embedding/StatefulPartitionedCall=private__token_and_position_embedding/StatefulPartitionedCall2h
2private__transformer_block/StatefulPartitionedCall2private__transformer_block/StatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

·
$__inference_model_layer_call_fn_5911
input_1
unknown:	 
	unknown_0:	Æ 
	unknown_1:  
	unknown_2: 
	unknown_3:  
	unknown_4: 
	unknown_5:  
	unknown_6: 
	unknown_7:  
	unknown_8: 
	unknown_9: 

unknown_10: 

unknown_11:  

unknown_12: 

unknown_13:  

unknown_14: 

unknown_15: 

unknown_16: 

unknown_17: 

unknown_18:

unknown_19:

unknown_20:
identity¢StatefulPartitionedCallä
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
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*8
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_5815o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1


ò
A__inference_dense_7_layer_call_and_return_conditional_losses_5246

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¼

&__inference_dense_7_layer_call_fn_7471

inputs
unknown:
	unknown_0:
identity¢StatefulPartitionedCallÖ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dense_7_layer_call_and_return_conditional_losses_5246o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

D
(__inference_dropout_3_layer_call_fn_7440

inputs
identity®
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dropout_3_layer_call_and_return_conditional_losses_5233`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

¦
D__inference_sequential_layer_call_and_return_conditional_losses_4861
dense_4_input
dense_4_4850:  
dense_4_4852: 
dense_5_4855:  
dense_5_4857: 
identity¢dense_4/StatefulPartitionedCall¢dense_5/StatefulPartitionedCallò
dense_4/StatefulPartitionedCallStatefulPartitionedCalldense_4_inputdense_4_4850dense_4_4852*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dense_4_layer_call_and_return_conditional_losses_4720
dense_5/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0dense_5_4855dense_5_4857*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dense_5_layer_call_and_return_conditional_losses_4756|
IdentityIdentity(dense_5/StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
NoOpNoOp ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ : : : : 2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall:[ W
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
'
_user_specified_namedense_4_input
Ö
a
C__inference_dropout_3_layer_call_and_return_conditional_losses_7450

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
÷
µ
"__inference_signature_wrapper_6078
input_1
unknown:	 
	unknown_0:	Æ 
	unknown_1:  
	unknown_2: 
	unknown_3:  
	unknown_4: 
	unknown_5:  
	unknown_6: 
	unknown_7:  
	unknown_8: 
	unknown_9: 

unknown_10: 

unknown_11:  

unknown_12: 

unknown_13:  

unknown_14: 

unknown_15: 

unknown_16: 

unknown_17: 

unknown_18:

unknown_19:

unknown_20:
identity¢StatefulPartitionedCallÄ
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
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*8
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *(
f#R!
__inference__wrapped_model_4682o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1


ò
A__inference_dense_6_layer_call_and_return_conditional_losses_5222

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
¿
Ó
)__inference_sequential_layer_call_fn_4847
dense_4_input
unknown:  
	unknown_0: 
	unknown_1:  
	unknown_2: 
identity¢StatefulPartitionedCallÿ
StatefulPartitionedCallStatefulPartitionedCalldense_4_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_4823t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
'
_user_specified_namedense_4_input
í
a
(__inference_dropout_2_layer_call_fn_7398

inputs
identity¢StatefulPartitionedCall¾
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dropout_2_layer_call_and_return_conditional_losses_5363o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Î
ø
A__inference_dense_5_layer_call_and_return_conditional_losses_7701

inputs3
!tensordot_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢Tensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:  *
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
value	B : »
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
value	B : ¿
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
:ÿÿÿÿÿÿÿÿÿ 
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ [
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: Y
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
:ÿÿÿÿÿÿÿÿÿ r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ d
IdentityIdentityBiasAdd:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Ñ
«
T__inference_private__transformer_block_layer_call_and_return_conditional_losses_5664

inputs\
Jprivate__multi_head_self_attention_dense_tensordot_readvariableop_resource:  V
Hprivate__multi_head_self_attention_dense_biasadd_readvariableop_resource: ^
Lprivate__multi_head_self_attention_dense_1_tensordot_readvariableop_resource:  X
Jprivate__multi_head_self_attention_dense_1_biasadd_readvariableop_resource: ^
Lprivate__multi_head_self_attention_dense_2_tensordot_readvariableop_resource:  X
Jprivate__multi_head_self_attention_dense_2_biasadd_readvariableop_resource: ^
Lprivate__multi_head_self_attention_dense_3_tensordot_readvariableop_resource:  X
Jprivate__multi_head_self_attention_dense_3_biasadd_readvariableop_resource: G
9layer_normalization_batchnorm_mul_readvariableop_resource: C
5layer_normalization_batchnorm_readvariableop_resource: F
4sequential_dense_4_tensordot_readvariableop_resource:  @
2sequential_dense_4_biasadd_readvariableop_resource: F
4sequential_dense_5_tensordot_readvariableop_resource:  @
2sequential_dense_5_biasadd_readvariableop_resource: I
;layer_normalization_1_batchnorm_mul_readvariableop_resource: E
7layer_normalization_1_batchnorm_readvariableop_resource: 
identity¢,layer_normalization/batchnorm/ReadVariableOp¢0layer_normalization/batchnorm/mul/ReadVariableOp¢.layer_normalization_1/batchnorm/ReadVariableOp¢2layer_normalization_1/batchnorm/mul/ReadVariableOp¢?private__multi_head_self_attention/dense/BiasAdd/ReadVariableOp¢Aprivate__multi_head_self_attention/dense/Tensordot/ReadVariableOp¢Aprivate__multi_head_self_attention/dense_1/BiasAdd/ReadVariableOp¢Cprivate__multi_head_self_attention/dense_1/Tensordot/ReadVariableOp¢Aprivate__multi_head_self_attention/dense_2/BiasAdd/ReadVariableOp¢Cprivate__multi_head_self_attention/dense_2/Tensordot/ReadVariableOp¢Aprivate__multi_head_self_attention/dense_3/BiasAdd/ReadVariableOp¢Cprivate__multi_head_self_attention/dense_3/Tensordot/ReadVariableOp¢)sequential/dense_4/BiasAdd/ReadVariableOp¢+sequential/dense_4/Tensordot/ReadVariableOp¢)sequential/dense_5/BiasAdd/ReadVariableOp¢+sequential/dense_5/Tensordot/ReadVariableOp^
(private__multi_head_self_attention/ShapeShapeinputs*
T0*
_output_shapes
:
6private__multi_head_self_attention/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
8private__multi_head_self_attention/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
8private__multi_head_self_attention/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
0private__multi_head_self_attention/strided_sliceStridedSlice1private__multi_head_self_attention/Shape:output:0?private__multi_head_self_attention/strided_slice/stack:output:0Aprivate__multi_head_self_attention/strided_slice/stack_1:output:0Aprivate__multi_head_self_attention/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskÌ
Aprivate__multi_head_self_attention/dense/Tensordot/ReadVariableOpReadVariableOpJprivate__multi_head_self_attention_dense_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype0
7private__multi_head_self_attention/dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:
7private__multi_head_self_attention/dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       n
8private__multi_head_self_attention/dense/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:
@private__multi_head_self_attention/dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ß
;private__multi_head_self_attention/dense/Tensordot/GatherV2GatherV2Aprivate__multi_head_self_attention/dense/Tensordot/Shape:output:0@private__multi_head_self_attention/dense/Tensordot/free:output:0Iprivate__multi_head_self_attention/dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:
Bprivate__multi_head_self_attention/dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ã
=private__multi_head_self_attention/dense/Tensordot/GatherV2_1GatherV2Aprivate__multi_head_self_attention/dense/Tensordot/Shape:output:0@private__multi_head_self_attention/dense/Tensordot/axes:output:0Kprivate__multi_head_self_attention/dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:
8private__multi_head_self_attention/dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: é
7private__multi_head_self_attention/dense/Tensordot/ProdProdDprivate__multi_head_self_attention/dense/Tensordot/GatherV2:output:0Aprivate__multi_head_self_attention/dense/Tensordot/Const:output:0*
T0*
_output_shapes
: 
:private__multi_head_self_attention/dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ï
9private__multi_head_self_attention/dense/Tensordot/Prod_1ProdFprivate__multi_head_self_attention/dense/Tensordot/GatherV2_1:output:0Cprivate__multi_head_self_attention/dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 
>private__multi_head_self_attention/dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : À
9private__multi_head_self_attention/dense/Tensordot/concatConcatV2@private__multi_head_self_attention/dense/Tensordot/free:output:0@private__multi_head_self_attention/dense/Tensordot/axes:output:0Gprivate__multi_head_self_attention/dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:ô
8private__multi_head_self_attention/dense/Tensordot/stackPack@private__multi_head_self_attention/dense/Tensordot/Prod:output:0Bprivate__multi_head_self_attention/dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:Ì
<private__multi_head_self_attention/dense/Tensordot/transpose	TransposeinputsBprivate__multi_head_self_attention/dense/Tensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
:private__multi_head_self_attention/dense/Tensordot/ReshapeReshape@private__multi_head_self_attention/dense/Tensordot/transpose:y:0Aprivate__multi_head_self_attention/dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
9private__multi_head_self_attention/dense/Tensordot/MatMulMatMulCprivate__multi_head_self_attention/dense/Tensordot/Reshape:output:0Iprivate__multi_head_self_attention/dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
:private__multi_head_self_attention/dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 
@private__multi_head_self_attention/dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ë
;private__multi_head_self_attention/dense/Tensordot/concat_1ConcatV2Dprivate__multi_head_self_attention/dense/Tensordot/GatherV2:output:0Cprivate__multi_head_self_attention/dense/Tensordot/Const_2:output:0Iprivate__multi_head_self_attention/dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:ÿ
2private__multi_head_self_attention/dense/TensordotReshapeCprivate__multi_head_self_attention/dense/Tensordot/MatMul:product:0Dprivate__multi_head_self_attention/dense/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ä
?private__multi_head_self_attention/dense/BiasAdd/ReadVariableOpReadVariableOpHprivate__multi_head_self_attention_dense_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0ø
0private__multi_head_self_attention/dense/BiasAddBiasAdd;private__multi_head_self_attention/dense/Tensordot:output:0Gprivate__multi_head_self_attention/dense/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ð
Cprivate__multi_head_self_attention/dense_1/Tensordot/ReadVariableOpReadVariableOpLprivate__multi_head_self_attention_dense_1_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype0
9private__multi_head_self_attention/dense_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:
9private__multi_head_self_attention/dense_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       p
:private__multi_head_self_attention/dense_1/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:
Bprivate__multi_head_self_attention/dense_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ç
=private__multi_head_self_attention/dense_1/Tensordot/GatherV2GatherV2Cprivate__multi_head_self_attention/dense_1/Tensordot/Shape:output:0Bprivate__multi_head_self_attention/dense_1/Tensordot/free:output:0Kprivate__multi_head_self_attention/dense_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:
Dprivate__multi_head_self_attention/dense_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ë
?private__multi_head_self_attention/dense_1/Tensordot/GatherV2_1GatherV2Cprivate__multi_head_self_attention/dense_1/Tensordot/Shape:output:0Bprivate__multi_head_self_attention/dense_1/Tensordot/axes:output:0Mprivate__multi_head_self_attention/dense_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:
:private__multi_head_self_attention/dense_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ï
9private__multi_head_self_attention/dense_1/Tensordot/ProdProdFprivate__multi_head_self_attention/dense_1/Tensordot/GatherV2:output:0Cprivate__multi_head_self_attention/dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: 
<private__multi_head_self_attention/dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: õ
;private__multi_head_self_attention/dense_1/Tensordot/Prod_1ProdHprivate__multi_head_self_attention/dense_1/Tensordot/GatherV2_1:output:0Eprivate__multi_head_self_attention/dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 
@private__multi_head_self_attention/dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : È
;private__multi_head_self_attention/dense_1/Tensordot/concatConcatV2Bprivate__multi_head_self_attention/dense_1/Tensordot/free:output:0Bprivate__multi_head_self_attention/dense_1/Tensordot/axes:output:0Iprivate__multi_head_self_attention/dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:ú
:private__multi_head_self_attention/dense_1/Tensordot/stackPackBprivate__multi_head_self_attention/dense_1/Tensordot/Prod:output:0Dprivate__multi_head_self_attention/dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:Ð
>private__multi_head_self_attention/dense_1/Tensordot/transpose	TransposeinputsDprivate__multi_head_self_attention/dense_1/Tensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
<private__multi_head_self_attention/dense_1/Tensordot/ReshapeReshapeBprivate__multi_head_self_attention/dense_1/Tensordot/transpose:y:0Cprivate__multi_head_self_attention/dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
;private__multi_head_self_attention/dense_1/Tensordot/MatMulMatMulEprivate__multi_head_self_attention/dense_1/Tensordot/Reshape:output:0Kprivate__multi_head_self_attention/dense_1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
<private__multi_head_self_attention/dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 
Bprivate__multi_head_self_attention/dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ó
=private__multi_head_self_attention/dense_1/Tensordot/concat_1ConcatV2Fprivate__multi_head_self_attention/dense_1/Tensordot/GatherV2:output:0Eprivate__multi_head_self_attention/dense_1/Tensordot/Const_2:output:0Kprivate__multi_head_self_attention/dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
4private__multi_head_self_attention/dense_1/TensordotReshapeEprivate__multi_head_self_attention/dense_1/Tensordot/MatMul:product:0Fprivate__multi_head_self_attention/dense_1/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ È
Aprivate__multi_head_self_attention/dense_1/BiasAdd/ReadVariableOpReadVariableOpJprivate__multi_head_self_attention_dense_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0þ
2private__multi_head_self_attention/dense_1/BiasAddBiasAdd=private__multi_head_self_attention/dense_1/Tensordot:output:0Iprivate__multi_head_self_attention/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ð
Cprivate__multi_head_self_attention/dense_2/Tensordot/ReadVariableOpReadVariableOpLprivate__multi_head_self_attention_dense_2_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype0
9private__multi_head_self_attention/dense_2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:
9private__multi_head_self_attention/dense_2/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       p
:private__multi_head_self_attention/dense_2/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:
Bprivate__multi_head_self_attention/dense_2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ç
=private__multi_head_self_attention/dense_2/Tensordot/GatherV2GatherV2Cprivate__multi_head_self_attention/dense_2/Tensordot/Shape:output:0Bprivate__multi_head_self_attention/dense_2/Tensordot/free:output:0Kprivate__multi_head_self_attention/dense_2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:
Dprivate__multi_head_self_attention/dense_2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ë
?private__multi_head_self_attention/dense_2/Tensordot/GatherV2_1GatherV2Cprivate__multi_head_self_attention/dense_2/Tensordot/Shape:output:0Bprivate__multi_head_self_attention/dense_2/Tensordot/axes:output:0Mprivate__multi_head_self_attention/dense_2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:
:private__multi_head_self_attention/dense_2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ï
9private__multi_head_self_attention/dense_2/Tensordot/ProdProdFprivate__multi_head_self_attention/dense_2/Tensordot/GatherV2:output:0Cprivate__multi_head_self_attention/dense_2/Tensordot/Const:output:0*
T0*
_output_shapes
: 
<private__multi_head_self_attention/dense_2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: õ
;private__multi_head_self_attention/dense_2/Tensordot/Prod_1ProdHprivate__multi_head_self_attention/dense_2/Tensordot/GatherV2_1:output:0Eprivate__multi_head_self_attention/dense_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 
@private__multi_head_self_attention/dense_2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : È
;private__multi_head_self_attention/dense_2/Tensordot/concatConcatV2Bprivate__multi_head_self_attention/dense_2/Tensordot/free:output:0Bprivate__multi_head_self_attention/dense_2/Tensordot/axes:output:0Iprivate__multi_head_self_attention/dense_2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:ú
:private__multi_head_self_attention/dense_2/Tensordot/stackPackBprivate__multi_head_self_attention/dense_2/Tensordot/Prod:output:0Dprivate__multi_head_self_attention/dense_2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:Ð
>private__multi_head_self_attention/dense_2/Tensordot/transpose	TransposeinputsDprivate__multi_head_self_attention/dense_2/Tensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
<private__multi_head_self_attention/dense_2/Tensordot/ReshapeReshapeBprivate__multi_head_self_attention/dense_2/Tensordot/transpose:y:0Cprivate__multi_head_self_attention/dense_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
;private__multi_head_self_attention/dense_2/Tensordot/MatMulMatMulEprivate__multi_head_self_attention/dense_2/Tensordot/Reshape:output:0Kprivate__multi_head_self_attention/dense_2/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
<private__multi_head_self_attention/dense_2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 
Bprivate__multi_head_self_attention/dense_2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ó
=private__multi_head_self_attention/dense_2/Tensordot/concat_1ConcatV2Fprivate__multi_head_self_attention/dense_2/Tensordot/GatherV2:output:0Eprivate__multi_head_self_attention/dense_2/Tensordot/Const_2:output:0Kprivate__multi_head_self_attention/dense_2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
4private__multi_head_self_attention/dense_2/TensordotReshapeEprivate__multi_head_self_attention/dense_2/Tensordot/MatMul:product:0Fprivate__multi_head_self_attention/dense_2/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ È
Aprivate__multi_head_self_attention/dense_2/BiasAdd/ReadVariableOpReadVariableOpJprivate__multi_head_self_attention_dense_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0þ
2private__multi_head_self_attention/dense_2/BiasAddBiasAdd=private__multi_head_self_attention/dense_2/Tensordot:output:0Iprivate__multi_head_self_attention/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ }
2private__multi_head_self_attention/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿt
2private__multi_head_self_attention/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :t
2private__multi_head_self_attention/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :Ø
0private__multi_head_self_attention/Reshape/shapePack9private__multi_head_self_attention/strided_slice:output:0;private__multi_head_self_attention/Reshape/shape/1:output:0;private__multi_head_self_attention/Reshape/shape/2:output:0;private__multi_head_self_attention/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:î
*private__multi_head_self_attention/ReshapeReshape9private__multi_head_self_attention/dense/BiasAdd:output:09private__multi_head_self_attention/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
1private__multi_head_self_attention/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             í
,private__multi_head_self_attention/transpose	Transpose3private__multi_head_self_attention/Reshape:output:0:private__multi_head_self_attention/transpose/perm:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
4private__multi_head_self_attention/Reshape_1/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿv
4private__multi_head_self_attention/Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :v
4private__multi_head_self_attention/Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :à
2private__multi_head_self_attention/Reshape_1/shapePack9private__multi_head_self_attention/strided_slice:output:0=private__multi_head_self_attention/Reshape_1/shape/1:output:0=private__multi_head_self_attention/Reshape_1/shape/2:output:0=private__multi_head_self_attention/Reshape_1/shape/3:output:0*
N*
T0*
_output_shapes
:ô
,private__multi_head_self_attention/Reshape_1Reshape;private__multi_head_self_attention/dense_1/BiasAdd:output:0;private__multi_head_self_attention/Reshape_1/shape:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
3private__multi_head_self_attention/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             ó
.private__multi_head_self_attention/transpose_1	Transpose5private__multi_head_self_attention/Reshape_1:output:0<private__multi_head_self_attention/transpose_1/perm:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
4private__multi_head_self_attention/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿv
4private__multi_head_self_attention/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :v
4private__multi_head_self_attention/Reshape_2/shape/3Const*
_output_shapes
: *
dtype0*
value	B :à
2private__multi_head_self_attention/Reshape_2/shapePack9private__multi_head_self_attention/strided_slice:output:0=private__multi_head_self_attention/Reshape_2/shape/1:output:0=private__multi_head_self_attention/Reshape_2/shape/2:output:0=private__multi_head_self_attention/Reshape_2/shape/3:output:0*
N*
T0*
_output_shapes
:ô
,private__multi_head_self_attention/Reshape_2Reshape;private__multi_head_self_attention/dense_2/BiasAdd:output:0;private__multi_head_self_attention/Reshape_2/shape:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
3private__multi_head_self_attention/transpose_2/permConst*
_output_shapes
:*
dtype0*%
valueB"             ó
.private__multi_head_self_attention/transpose_2	Transpose5private__multi_head_self_attention/Reshape_2:output:0<private__multi_head_self_attention/transpose_2/perm:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿù
)private__multi_head_self_attention/MatMulBatchMatMulV20private__multi_head_self_attention/transpose:y:02private__multi_head_self_attention/transpose_1:y:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
adj_y(
*private__multi_head_self_attention/Shape_1Shape2private__multi_head_self_attention/transpose_1:y:0*
T0*
_output_shapes
:
8private__multi_head_self_attention/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ
:private__multi_head_self_attention/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
:private__multi_head_self_attention/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
2private__multi_head_self_attention/strided_slice_1StridedSlice3private__multi_head_self_attention/Shape_1:output:0Aprivate__multi_head_self_attention/strided_slice_1/stack:output:0Cprivate__multi_head_self_attention/strided_slice_1/stack_1:output:0Cprivate__multi_head_self_attention/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
'private__multi_head_self_attention/CastCast;private__multi_head_self_attention/strided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: }
'private__multi_head_self_attention/SqrtSqrt+private__multi_head_self_attention/Cast:y:0*
T0*
_output_shapes
: â
*private__multi_head_self_attention/truedivRealDiv2private__multi_head_self_attention/MatMul:output:0+private__multi_head_self_attention/Sqrt:y:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ±
*private__multi_head_self_attention/SoftmaxSoftmax.private__multi_head_self_attention/truediv:z:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿé
+private__multi_head_self_attention/MatMul_1BatchMatMulV24private__multi_head_self_attention/Softmax:softmax:02private__multi_head_self_attention/transpose_2:y:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
3private__multi_head_self_attention/transpose_3/permConst*
_output_shapes
:*
dtype0*%
valueB"             ò
.private__multi_head_self_attention/transpose_3	Transpose4private__multi_head_self_attention/MatMul_1:output:0<private__multi_head_self_attention/transpose_3/perm:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
4private__multi_head_self_attention/Reshape_3/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿv
4private__multi_head_self_attention/Reshape_3/shape/2Const*
_output_shapes
: *
dtype0*
value	B : ¡
2private__multi_head_self_attention/Reshape_3/shapePack9private__multi_head_self_attention/strided_slice:output:0=private__multi_head_self_attention/Reshape_3/shape/1:output:0=private__multi_head_self_attention/Reshape_3/shape/2:output:0*
N*
T0*
_output_shapes
:ç
,private__multi_head_self_attention/Reshape_3Reshape2private__multi_head_self_attention/transpose_3:y:0;private__multi_head_self_attention/Reshape_3/shape:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ Ð
Cprivate__multi_head_self_attention/dense_3/Tensordot/ReadVariableOpReadVariableOpLprivate__multi_head_self_attention_dense_3_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype0
9private__multi_head_self_attention/dense_3/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:
9private__multi_head_self_attention/dense_3/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       
:private__multi_head_self_attention/dense_3/Tensordot/ShapeShape5private__multi_head_self_attention/Reshape_3:output:0*
T0*
_output_shapes
:
Bprivate__multi_head_self_attention/dense_3/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ç
=private__multi_head_self_attention/dense_3/Tensordot/GatherV2GatherV2Cprivate__multi_head_self_attention/dense_3/Tensordot/Shape:output:0Bprivate__multi_head_self_attention/dense_3/Tensordot/free:output:0Kprivate__multi_head_self_attention/dense_3/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:
Dprivate__multi_head_self_attention/dense_3/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ë
?private__multi_head_self_attention/dense_3/Tensordot/GatherV2_1GatherV2Cprivate__multi_head_self_attention/dense_3/Tensordot/Shape:output:0Bprivate__multi_head_self_attention/dense_3/Tensordot/axes:output:0Mprivate__multi_head_self_attention/dense_3/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:
:private__multi_head_self_attention/dense_3/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ï
9private__multi_head_self_attention/dense_3/Tensordot/ProdProdFprivate__multi_head_self_attention/dense_3/Tensordot/GatherV2:output:0Cprivate__multi_head_self_attention/dense_3/Tensordot/Const:output:0*
T0*
_output_shapes
: 
<private__multi_head_self_attention/dense_3/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: õ
;private__multi_head_self_attention/dense_3/Tensordot/Prod_1ProdHprivate__multi_head_self_attention/dense_3/Tensordot/GatherV2_1:output:0Eprivate__multi_head_self_attention/dense_3/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 
@private__multi_head_self_attention/dense_3/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : È
;private__multi_head_self_attention/dense_3/Tensordot/concatConcatV2Bprivate__multi_head_self_attention/dense_3/Tensordot/free:output:0Bprivate__multi_head_self_attention/dense_3/Tensordot/axes:output:0Iprivate__multi_head_self_attention/dense_3/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:ú
:private__multi_head_self_attention/dense_3/Tensordot/stackPackBprivate__multi_head_self_attention/dense_3/Tensordot/Prod:output:0Dprivate__multi_head_self_attention/dense_3/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:
>private__multi_head_self_attention/dense_3/Tensordot/transpose	Transpose5private__multi_head_self_attention/Reshape_3:output:0Dprivate__multi_head_self_attention/dense_3/Tensordot/concat:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
<private__multi_head_self_attention/dense_3/Tensordot/ReshapeReshapeBprivate__multi_head_self_attention/dense_3/Tensordot/transpose:y:0Cprivate__multi_head_self_attention/dense_3/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
;private__multi_head_self_attention/dense_3/Tensordot/MatMulMatMulEprivate__multi_head_self_attention/dense_3/Tensordot/Reshape:output:0Kprivate__multi_head_self_attention/dense_3/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
<private__multi_head_self_attention/dense_3/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 
Bprivate__multi_head_self_attention/dense_3/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ó
=private__multi_head_self_attention/dense_3/Tensordot/concat_1ConcatV2Fprivate__multi_head_self_attention/dense_3/Tensordot/GatherV2:output:0Eprivate__multi_head_self_attention/dense_3/Tensordot/Const_2:output:0Kprivate__multi_head_self_attention/dense_3/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
4private__multi_head_self_attention/dense_3/TensordotReshapeEprivate__multi_head_self_attention/dense_3/Tensordot/MatMul:product:0Fprivate__multi_head_self_attention/dense_3/Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ È
Aprivate__multi_head_self_attention/dense_3/BiasAdd/ReadVariableOpReadVariableOpJprivate__multi_head_self_attention_dense_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
2private__multi_head_self_attention/dense_3/BiasAddBiasAdd=private__multi_head_self_attention/dense_3/Tensordot:output:0Iprivate__multi_head_self_attention/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ Z
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?¶
dropout/dropout/MulMul;private__multi_head_self_attention/dense_3/BiasAdd:output:0dropout/dropout/Const:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
dropout/dropout/ShapeShape;private__multi_head_self_attention/dense_3/BiasAdd:output:0*
T0*
_output_shapes
:©
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *
dtype0c
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=Ë
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ f
addAddV2inputsdropout/dropout/Mul_1:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ |
2layer_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:¶
 layer_normalization/moments/meanMeanadd:z:0;layer_normalization/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(
(layer_normalization/moments/StopGradientStopGradient)layer_normalization/moments/mean:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿµ
-layer_normalization/moments/SquaredDifferenceSquaredDifferenceadd:z:01layer_normalization/moments/StopGradient:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
6layer_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:è
$layer_normalization/moments/varianceMean1layer_normalization/moments/SquaredDifference:z:0?layer_normalization/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(h
#layer_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½75¾
!layer_normalization/batchnorm/addAddV2-layer_normalization/moments/variance:output:0,layer_normalization/batchnorm/add/y:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#layer_normalization/batchnorm/RsqrtRsqrt%layer_normalization/batchnorm/add:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¦
0layer_normalization/batchnorm/mul/ReadVariableOpReadVariableOp9layer_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype0Â
!layer_normalization/batchnorm/mulMul'layer_normalization/batchnorm/Rsqrt:y:08layer_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
#layer_normalization/batchnorm/mul_1Muladd:z:0%layer_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ³
#layer_normalization/batchnorm/mul_2Mul)layer_normalization/moments/mean:output:0%layer_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
,layer_normalization/batchnorm/ReadVariableOpReadVariableOp5layer_normalization_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype0¾
!layer_normalization/batchnorm/subSub4layer_normalization/batchnorm/ReadVariableOp:value:0'layer_normalization/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ³
#layer_normalization/batchnorm/add_1AddV2'layer_normalization/batchnorm/mul_1:z:0%layer_normalization/batchnorm/sub:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
+sequential/dense_4/Tensordot/ReadVariableOpReadVariableOp4sequential_dense_4_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype0k
!sequential/dense_4/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:r
!sequential/dense_4/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       y
"sequential/dense_4/Tensordot/ShapeShape'layer_normalization/batchnorm/add_1:z:0*
T0*
_output_shapes
:l
*sequential/dense_4/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
%sequential/dense_4/Tensordot/GatherV2GatherV2+sequential/dense_4/Tensordot/Shape:output:0*sequential/dense_4/Tensordot/free:output:03sequential/dense_4/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:n
,sequential/dense_4/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
'sequential/dense_4/Tensordot/GatherV2_1GatherV2+sequential/dense_4/Tensordot/Shape:output:0*sequential/dense_4/Tensordot/axes:output:05sequential/dense_4/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:l
"sequential/dense_4/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: §
!sequential/dense_4/Tensordot/ProdProd.sequential/dense_4/Tensordot/GatherV2:output:0+sequential/dense_4/Tensordot/Const:output:0*
T0*
_output_shapes
: n
$sequential/dense_4/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ­
#sequential/dense_4/Tensordot/Prod_1Prod0sequential/dense_4/Tensordot/GatherV2_1:output:0-sequential/dense_4/Tensordot/Const_1:output:0*
T0*
_output_shapes
: j
(sequential/dense_4/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : è
#sequential/dense_4/Tensordot/concatConcatV2*sequential/dense_4/Tensordot/free:output:0*sequential/dense_4/Tensordot/axes:output:01sequential/dense_4/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:²
"sequential/dense_4/Tensordot/stackPack*sequential/dense_4/Tensordot/Prod:output:0,sequential/dense_4/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:Á
&sequential/dense_4/Tensordot/transpose	Transpose'layer_normalization/batchnorm/add_1:z:0,sequential/dense_4/Tensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ã
$sequential/dense_4/Tensordot/ReshapeReshape*sequential/dense_4/Tensordot/transpose:y:0+sequential/dense_4/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÃ
#sequential/dense_4/Tensordot/MatMulMatMul-sequential/dense_4/Tensordot/Reshape:output:03sequential/dense_4/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ n
$sequential/dense_4/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: l
*sequential/dense_4/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ó
%sequential/dense_4/Tensordot/concat_1ConcatV2.sequential/dense_4/Tensordot/GatherV2:output:0-sequential/dense_4/Tensordot/Const_2:output:03sequential/dense_4/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:½
sequential/dense_4/TensordotReshape-sequential/dense_4/Tensordot/MatMul:product:0.sequential/dense_4/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
)sequential/dense_4/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0¶
sequential/dense_4/BiasAddBiasAdd%sequential/dense_4/Tensordot:output:01sequential/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ {
sequential/dense_4/ReluRelu#sequential/dense_4/BiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
+sequential/dense_5/Tensordot/ReadVariableOpReadVariableOp4sequential_dense_5_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype0k
!sequential/dense_5/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:r
!sequential/dense_5/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       w
"sequential/dense_5/Tensordot/ShapeShape%sequential/dense_4/Relu:activations:0*
T0*
_output_shapes
:l
*sequential/dense_5/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
%sequential/dense_5/Tensordot/GatherV2GatherV2+sequential/dense_5/Tensordot/Shape:output:0*sequential/dense_5/Tensordot/free:output:03sequential/dense_5/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:n
,sequential/dense_5/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
'sequential/dense_5/Tensordot/GatherV2_1GatherV2+sequential/dense_5/Tensordot/Shape:output:0*sequential/dense_5/Tensordot/axes:output:05sequential/dense_5/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:l
"sequential/dense_5/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: §
!sequential/dense_5/Tensordot/ProdProd.sequential/dense_5/Tensordot/GatherV2:output:0+sequential/dense_5/Tensordot/Const:output:0*
T0*
_output_shapes
: n
$sequential/dense_5/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ­
#sequential/dense_5/Tensordot/Prod_1Prod0sequential/dense_5/Tensordot/GatherV2_1:output:0-sequential/dense_5/Tensordot/Const_1:output:0*
T0*
_output_shapes
: j
(sequential/dense_5/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : è
#sequential/dense_5/Tensordot/concatConcatV2*sequential/dense_5/Tensordot/free:output:0*sequential/dense_5/Tensordot/axes:output:01sequential/dense_5/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:²
"sequential/dense_5/Tensordot/stackPack*sequential/dense_5/Tensordot/Prod:output:0,sequential/dense_5/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:¿
&sequential/dense_5/Tensordot/transpose	Transpose%sequential/dense_4/Relu:activations:0,sequential/dense_5/Tensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ã
$sequential/dense_5/Tensordot/ReshapeReshape*sequential/dense_5/Tensordot/transpose:y:0+sequential/dense_5/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÃ
#sequential/dense_5/Tensordot/MatMulMatMul-sequential/dense_5/Tensordot/Reshape:output:03sequential/dense_5/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ n
$sequential/dense_5/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: l
*sequential/dense_5/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ó
%sequential/dense_5/Tensordot/concat_1ConcatV2.sequential/dense_5/Tensordot/GatherV2:output:0-sequential/dense_5/Tensordot/Const_2:output:03sequential/dense_5/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:½
sequential/dense_5/TensordotReshape-sequential/dense_5/Tensordot/MatMul:product:0.sequential/dense_5/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
)sequential/dense_5/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0¶
sequential/dense_5/BiasAddBiasAdd%sequential/dense_5/Tensordot:output:01sequential/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ \
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?
dropout_1/dropout/MulMul#sequential/dense_5/BiasAdd:output:0 dropout_1/dropout/Const:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ j
dropout_1/dropout/ShapeShape#sequential/dense_5/BiasAdd:output:0*
T0*
_output_shapes
:¥
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
dtype0e
 dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=É
dropout_1/dropout/GreaterEqualGreaterEqual7dropout_1/dropout/random_uniform/RandomUniform:output:0)dropout_1/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
dropout_1/dropout/Mul_1Muldropout_1/dropout/Mul:z:0dropout_1/dropout/Cast:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
add_1AddV2'layer_normalization/batchnorm/add_1:z:0dropout_1/dropout/Mul_1:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ~
4layer_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:¼
"layer_normalization_1/moments/meanMean	add_1:z:0=layer_normalization_1/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(
*layer_normalization_1/moments/StopGradientStopGradient+layer_normalization_1/moments/mean:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ»
/layer_normalization_1/moments/SquaredDifferenceSquaredDifference	add_1:z:03layer_normalization_1/moments/StopGradient:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
8layer_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:î
&layer_normalization_1/moments/varianceMean3layer_normalization_1/moments/SquaredDifference:z:0Alayer_normalization_1/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(j
%layer_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½75Ä
#layer_normalization_1/batchnorm/addAddV2/layer_normalization_1/moments/variance:output:0.layer_normalization_1/batchnorm/add/y:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%layer_normalization_1/batchnorm/RsqrtRsqrt'layer_normalization_1/batchnorm/add:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
2layer_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype0È
#layer_normalization_1/batchnorm/mulMul)layer_normalization_1/batchnorm/Rsqrt:y:0:layer_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
%layer_normalization_1/batchnorm/mul_1Mul	add_1:z:0'layer_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¹
%layer_normalization_1/batchnorm/mul_2Mul+layer_normalization_1/moments/mean:output:0'layer_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¢
.layer_normalization_1/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype0Ä
#layer_normalization_1/batchnorm/subSub6layer_normalization_1/batchnorm/ReadVariableOp:value:0)layer_normalization_1/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¹
%layer_normalization_1/batchnorm/add_1AddV2)layer_normalization_1/batchnorm/mul_1:z:0'layer_normalization_1/batchnorm/sub:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ }
IdentityIdentity)layer_normalization_1/batchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ æ
NoOpNoOp-^layer_normalization/batchnorm/ReadVariableOp1^layer_normalization/batchnorm/mul/ReadVariableOp/^layer_normalization_1/batchnorm/ReadVariableOp3^layer_normalization_1/batchnorm/mul/ReadVariableOp@^private__multi_head_self_attention/dense/BiasAdd/ReadVariableOpB^private__multi_head_self_attention/dense/Tensordot/ReadVariableOpB^private__multi_head_self_attention/dense_1/BiasAdd/ReadVariableOpD^private__multi_head_self_attention/dense_1/Tensordot/ReadVariableOpB^private__multi_head_self_attention/dense_2/BiasAdd/ReadVariableOpD^private__multi_head_self_attention/dense_2/Tensordot/ReadVariableOpB^private__multi_head_self_attention/dense_3/BiasAdd/ReadVariableOpD^private__multi_head_self_attention/dense_3/Tensordot/ReadVariableOp*^sequential/dense_4/BiasAdd/ReadVariableOp,^sequential/dense_4/Tensordot/ReadVariableOp*^sequential/dense_5/BiasAdd/ReadVariableOp,^sequential/dense_5/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:ÿÿÿÿÿÿÿÿÿ : : : : : : : : : : : : : : : : 2\
,layer_normalization/batchnorm/ReadVariableOp,layer_normalization/batchnorm/ReadVariableOp2d
0layer_normalization/batchnorm/mul/ReadVariableOp0layer_normalization/batchnorm/mul/ReadVariableOp2`
.layer_normalization_1/batchnorm/ReadVariableOp.layer_normalization_1/batchnorm/ReadVariableOp2h
2layer_normalization_1/batchnorm/mul/ReadVariableOp2layer_normalization_1/batchnorm/mul/ReadVariableOp2
?private__multi_head_self_attention/dense/BiasAdd/ReadVariableOp?private__multi_head_self_attention/dense/BiasAdd/ReadVariableOp2
Aprivate__multi_head_self_attention/dense/Tensordot/ReadVariableOpAprivate__multi_head_self_attention/dense/Tensordot/ReadVariableOp2
Aprivate__multi_head_self_attention/dense_1/BiasAdd/ReadVariableOpAprivate__multi_head_self_attention/dense_1/BiasAdd/ReadVariableOp2
Cprivate__multi_head_self_attention/dense_1/Tensordot/ReadVariableOpCprivate__multi_head_self_attention/dense_1/Tensordot/ReadVariableOp2
Aprivate__multi_head_self_attention/dense_2/BiasAdd/ReadVariableOpAprivate__multi_head_self_attention/dense_2/BiasAdd/ReadVariableOp2
Cprivate__multi_head_self_attention/dense_2/Tensordot/ReadVariableOpCprivate__multi_head_self_attention/dense_2/Tensordot/ReadVariableOp2
Aprivate__multi_head_self_attention/dense_3/BiasAdd/ReadVariableOpAprivate__multi_head_self_attention/dense_3/BiasAdd/ReadVariableOp2
Cprivate__multi_head_self_attention/dense_3/Tensordot/ReadVariableOpCprivate__multi_head_self_attention/dense_3/Tensordot/ReadVariableOp2V
)sequential/dense_4/BiasAdd/ReadVariableOp)sequential/dense_4/BiasAdd/ReadVariableOp2Z
+sequential/dense_4/Tensordot/ReadVariableOp+sequential/dense_4/Tensordot/ReadVariableOp2V
)sequential/dense_5/BiasAdd/ReadVariableOp)sequential/dense_5/BiasAdd/ReadVariableOp2Z
+sequential/dense_5/Tensordot/ReadVariableOp+sequential/dense_5/Tensordot/ReadVariableOp:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Ö
a
C__inference_dropout_3_layer_call_and_return_conditional_losses_5233

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ñ	
b
C__inference_dropout_3_layer_call_and_return_conditional_losses_5330

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=¦
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿo
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
÷
S
7__inference_global_average_pooling1d_layer_call_fn_7382

inputs
identityÆ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_4885i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¥Ô

?__inference_model_layer_call_and_return_conditional_losses_6768

inputsZ
Gprivate__token_and_position_embedding_embedding_1_embedding_lookup_6469:	 X
Eprivate__token_and_position_embedding_embedding_embedding_lookup_6475:	Æ w
eprivate__transformer_block_private__multi_head_self_attention_dense_tensordot_readvariableop_resource:  q
cprivate__transformer_block_private__multi_head_self_attention_dense_biasadd_readvariableop_resource: y
gprivate__transformer_block_private__multi_head_self_attention_dense_1_tensordot_readvariableop_resource:  s
eprivate__transformer_block_private__multi_head_self_attention_dense_1_biasadd_readvariableop_resource: y
gprivate__transformer_block_private__multi_head_self_attention_dense_2_tensordot_readvariableop_resource:  s
eprivate__transformer_block_private__multi_head_self_attention_dense_2_biasadd_readvariableop_resource: y
gprivate__transformer_block_private__multi_head_self_attention_dense_3_tensordot_readvariableop_resource:  s
eprivate__transformer_block_private__multi_head_self_attention_dense_3_biasadd_readvariableop_resource: b
Tprivate__transformer_block_layer_normalization_batchnorm_mul_readvariableop_resource: ^
Pprivate__transformer_block_layer_normalization_batchnorm_readvariableop_resource: a
Oprivate__transformer_block_sequential_dense_4_tensordot_readvariableop_resource:  [
Mprivate__transformer_block_sequential_dense_4_biasadd_readvariableop_resource: a
Oprivate__transformer_block_sequential_dense_5_tensordot_readvariableop_resource:  [
Mprivate__transformer_block_sequential_dense_5_biasadd_readvariableop_resource: d
Vprivate__transformer_block_layer_normalization_1_batchnorm_mul_readvariableop_resource: `
Rprivate__transformer_block_layer_normalization_1_batchnorm_readvariableop_resource: 8
&dense_6_matmul_readvariableop_resource: 5
'dense_6_biasadd_readvariableop_resource:8
&dense_7_matmul_readvariableop_resource:5
'dense_7_biasadd_readvariableop_resource:
identity¢dense_6/BiasAdd/ReadVariableOp¢dense_6/MatMul/ReadVariableOp¢dense_7/BiasAdd/ReadVariableOp¢dense_7/MatMul/ReadVariableOp¢@private__token_and_position_embedding/embedding/embedding_lookup¢Bprivate__token_and_position_embedding/embedding_1/embedding_lookup¢Gprivate__transformer_block/layer_normalization/batchnorm/ReadVariableOp¢Kprivate__transformer_block/layer_normalization/batchnorm/mul/ReadVariableOp¢Iprivate__transformer_block/layer_normalization_1/batchnorm/ReadVariableOp¢Mprivate__transformer_block/layer_normalization_1/batchnorm/mul/ReadVariableOp¢Zprivate__transformer_block/private__multi_head_self_attention/dense/BiasAdd/ReadVariableOp¢\private__transformer_block/private__multi_head_self_attention/dense/Tensordot/ReadVariableOp¢\private__transformer_block/private__multi_head_self_attention/dense_1/BiasAdd/ReadVariableOp¢^private__transformer_block/private__multi_head_self_attention/dense_1/Tensordot/ReadVariableOp¢\private__transformer_block/private__multi_head_self_attention/dense_2/BiasAdd/ReadVariableOp¢^private__transformer_block/private__multi_head_self_attention/dense_2/Tensordot/ReadVariableOp¢\private__transformer_block/private__multi_head_self_attention/dense_3/BiasAdd/ReadVariableOp¢^private__transformer_block/private__multi_head_self_attention/dense_3/Tensordot/ReadVariableOp¢Dprivate__transformer_block/sequential/dense_4/BiasAdd/ReadVariableOp¢Fprivate__transformer_block/sequential/dense_4/Tensordot/ReadVariableOp¢Dprivate__transformer_block/sequential/dense_5/BiasAdd/ReadVariableOp¢Fprivate__transformer_block/sequential/dense_5/Tensordot/ReadVariableOpa
+private__token_and_position_embedding/ShapeShapeinputs*
T0*
_output_shapes
:
9private__token_and_position_embedding/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ
;private__token_and_position_embedding/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
;private__token_and_position_embedding/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
3private__token_and_position_embedding/strided_sliceStridedSlice4private__token_and_position_embedding/Shape:output:0Bprivate__token_and_position_embedding/strided_slice/stack:output:0Dprivate__token_and_position_embedding/strided_slice/stack_1:output:0Dprivate__token_and_position_embedding/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masks
1private__token_and_position_embedding/range/startConst*
_output_shapes
: *
dtype0*
value	B : s
1private__token_and_position_embedding/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :
+private__token_and_position_embedding/rangeRange:private__token_and_position_embedding/range/start:output:0<private__token_and_position_embedding/strided_slice:output:0:private__token_and_position_embedding/range/delta:output:0*
_output_shapes	
:í
Bprivate__token_and_position_embedding/embedding_1/embedding_lookupResourceGatherGprivate__token_and_position_embedding_embedding_1_embedding_lookup_64694private__token_and_position_embedding/range:output:0*
Tindices0*Z
_classP
NLloc:@private__token_and_position_embedding/embedding_1/embedding_lookup/6469*
_output_shapes
:	 *
dtype0ª
Kprivate__token_and_position_embedding/embedding_1/embedding_lookup/IdentityIdentityKprivate__token_and_position_embedding/embedding_1/embedding_lookup:output:0*
T0*Z
_classP
NLloc:@private__token_and_position_embedding/embedding_1/embedding_lookup/6469*
_output_shapes
:	 Ù
Mprivate__token_and_position_embedding/embedding_1/embedding_lookup/Identity_1IdentityTprivate__token_and_position_embedding/embedding_1/embedding_lookup/Identity:output:0*
T0*
_output_shapes
:	 
4private__token_and_position_embedding/embedding/CastCastinputs*

DstT0*

SrcT0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿø
@private__token_and_position_embedding/embedding/embedding_lookupResourceGatherEprivate__token_and_position_embedding_embedding_embedding_lookup_64758private__token_and_position_embedding/embedding/Cast:y:0*
Tindices0*X
_classN
LJloc:@private__token_and_position_embedding/embedding/embedding_lookup/6475*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
dtype0±
Iprivate__token_and_position_embedding/embedding/embedding_lookup/IdentityIdentityIprivate__token_and_position_embedding/embedding/embedding_lookup:output:0*
T0*X
_classN
LJloc:@private__token_and_position_embedding/embedding/embedding_lookup/6475*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ â
Kprivate__token_and_position_embedding/embedding/embedding_lookup/Identity_1IdentityRprivate__token_and_position_embedding/embedding/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
)private__token_and_position_embedding/addAddV2Tprivate__token_and_position_embedding/embedding/embedding_lookup/Identity_1:output:0Vprivate__token_and_position_embedding/embedding_1/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
Cprivate__transformer_block/private__multi_head_self_attention/ShapeShape-private__token_and_position_embedding/add:z:0*
T0*
_output_shapes
:
Qprivate__transformer_block/private__multi_head_self_attention/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Sprivate__transformer_block/private__multi_head_self_attention/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Sprivate__transformer_block/private__multi_head_self_attention/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
Kprivate__transformer_block/private__multi_head_self_attention/strided_sliceStridedSliceLprivate__transformer_block/private__multi_head_self_attention/Shape:output:0Zprivate__transformer_block/private__multi_head_self_attention/strided_slice/stack:output:0\private__transformer_block/private__multi_head_self_attention/strided_slice/stack_1:output:0\private__transformer_block/private__multi_head_self_attention/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
\private__transformer_block/private__multi_head_self_attention/dense/Tensordot/ReadVariableOpReadVariableOpeprivate__transformer_block_private__multi_head_self_attention_dense_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype0
Rprivate__transformer_block/private__multi_head_self_attention/dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:£
Rprivate__transformer_block/private__multi_head_self_attention/dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       °
Sprivate__transformer_block/private__multi_head_self_attention/dense/Tensordot/ShapeShape-private__token_and_position_embedding/add:z:0*
T0*
_output_shapes
:
[private__transformer_block/private__multi_head_self_attention/dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : Ë
Vprivate__transformer_block/private__multi_head_self_attention/dense/Tensordot/GatherV2GatherV2\private__transformer_block/private__multi_head_self_attention/dense/Tensordot/Shape:output:0[private__transformer_block/private__multi_head_self_attention/dense/Tensordot/free:output:0dprivate__transformer_block/private__multi_head_self_attention/dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:
]private__transformer_block/private__multi_head_self_attention/dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ï
Xprivate__transformer_block/private__multi_head_self_attention/dense/Tensordot/GatherV2_1GatherV2\private__transformer_block/private__multi_head_self_attention/dense/Tensordot/Shape:output:0[private__transformer_block/private__multi_head_self_attention/dense/Tensordot/axes:output:0fprivate__transformer_block/private__multi_head_self_attention/dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:
Sprivate__transformer_block/private__multi_head_self_attention/dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: º
Rprivate__transformer_block/private__multi_head_self_attention/dense/Tensordot/ProdProd_private__transformer_block/private__multi_head_self_attention/dense/Tensordot/GatherV2:output:0\private__transformer_block/private__multi_head_self_attention/dense/Tensordot/Const:output:0*
T0*
_output_shapes
: 
Uprivate__transformer_block/private__multi_head_self_attention/dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: À
Tprivate__transformer_block/private__multi_head_self_attention/dense/Tensordot/Prod_1Prodaprivate__transformer_block/private__multi_head_self_attention/dense/Tensordot/GatherV2_1:output:0^private__transformer_block/private__multi_head_self_attention/dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 
Yprivate__transformer_block/private__multi_head_self_attention/dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ¬
Tprivate__transformer_block/private__multi_head_self_attention/dense/Tensordot/concatConcatV2[private__transformer_block/private__multi_head_self_attention/dense/Tensordot/free:output:0[private__transformer_block/private__multi_head_self_attention/dense/Tensordot/axes:output:0bprivate__transformer_block/private__multi_head_self_attention/dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:Å
Sprivate__transformer_block/private__multi_head_self_attention/dense/Tensordot/stackPack[private__transformer_block/private__multi_head_self_attention/dense/Tensordot/Prod:output:0]private__transformer_block/private__multi_head_self_attention/dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:©
Wprivate__transformer_block/private__multi_head_self_attention/dense/Tensordot/transpose	Transpose-private__token_and_position_embedding/add:z:0]private__transformer_block/private__multi_head_self_attention/dense/Tensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ö
Uprivate__transformer_block/private__multi_head_self_attention/dense/Tensordot/ReshapeReshape[private__transformer_block/private__multi_head_self_attention/dense/Tensordot/transpose:y:0\private__transformer_block/private__multi_head_self_attention/dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÖ
Tprivate__transformer_block/private__multi_head_self_attention/dense/Tensordot/MatMulMatMul^private__transformer_block/private__multi_head_self_attention/dense/Tensordot/Reshape:output:0dprivate__transformer_block/private__multi_head_self_attention/dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
Uprivate__transformer_block/private__multi_head_self_attention/dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 
[private__transformer_block/private__multi_head_self_attention/dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ·
Vprivate__transformer_block/private__multi_head_self_attention/dense/Tensordot/concat_1ConcatV2_private__transformer_block/private__multi_head_self_attention/dense/Tensordot/GatherV2:output:0^private__transformer_block/private__multi_head_self_attention/dense/Tensordot/Const_2:output:0dprivate__transformer_block/private__multi_head_self_attention/dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:Ð
Mprivate__transformer_block/private__multi_head_self_attention/dense/TensordotReshape^private__transformer_block/private__multi_head_self_attention/dense/Tensordot/MatMul:product:0_private__transformer_block/private__multi_head_self_attention/dense/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ú
Zprivate__transformer_block/private__multi_head_self_attention/dense/BiasAdd/ReadVariableOpReadVariableOpcprivate__transformer_block_private__multi_head_self_attention_dense_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0É
Kprivate__transformer_block/private__multi_head_self_attention/dense/BiasAddBiasAddVprivate__transformer_block/private__multi_head_self_attention/dense/Tensordot:output:0bprivate__transformer_block/private__multi_head_self_attention/dense/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
^private__transformer_block/private__multi_head_self_attention/dense_1/Tensordot/ReadVariableOpReadVariableOpgprivate__transformer_block_private__multi_head_self_attention_dense_1_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype0
Tprivate__transformer_block/private__multi_head_self_attention/dense_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:¥
Tprivate__transformer_block/private__multi_head_self_attention/dense_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       ²
Uprivate__transformer_block/private__multi_head_self_attention/dense_1/Tensordot/ShapeShape-private__token_and_position_embedding/add:z:0*
T0*
_output_shapes
:
]private__transformer_block/private__multi_head_self_attention/dense_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : Ó
Xprivate__transformer_block/private__multi_head_self_attention/dense_1/Tensordot/GatherV2GatherV2^private__transformer_block/private__multi_head_self_attention/dense_1/Tensordot/Shape:output:0]private__transformer_block/private__multi_head_self_attention/dense_1/Tensordot/free:output:0fprivate__transformer_block/private__multi_head_self_attention/dense_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:¡
_private__transformer_block/private__multi_head_self_attention/dense_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ×
Zprivate__transformer_block/private__multi_head_self_attention/dense_1/Tensordot/GatherV2_1GatherV2^private__transformer_block/private__multi_head_self_attention/dense_1/Tensordot/Shape:output:0]private__transformer_block/private__multi_head_self_attention/dense_1/Tensordot/axes:output:0hprivate__transformer_block/private__multi_head_self_attention/dense_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:
Uprivate__transformer_block/private__multi_head_self_attention/dense_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: À
Tprivate__transformer_block/private__multi_head_self_attention/dense_1/Tensordot/ProdProdaprivate__transformer_block/private__multi_head_self_attention/dense_1/Tensordot/GatherV2:output:0^private__transformer_block/private__multi_head_self_attention/dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: ¡
Wprivate__transformer_block/private__multi_head_self_attention/dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: Æ
Vprivate__transformer_block/private__multi_head_self_attention/dense_1/Tensordot/Prod_1Prodcprivate__transformer_block/private__multi_head_self_attention/dense_1/Tensordot/GatherV2_1:output:0`private__transformer_block/private__multi_head_self_attention/dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 
[private__transformer_block/private__multi_head_self_attention/dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ´
Vprivate__transformer_block/private__multi_head_self_attention/dense_1/Tensordot/concatConcatV2]private__transformer_block/private__multi_head_self_attention/dense_1/Tensordot/free:output:0]private__transformer_block/private__multi_head_self_attention/dense_1/Tensordot/axes:output:0dprivate__transformer_block/private__multi_head_self_attention/dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:Ë
Uprivate__transformer_block/private__multi_head_self_attention/dense_1/Tensordot/stackPack]private__transformer_block/private__multi_head_self_attention/dense_1/Tensordot/Prod:output:0_private__transformer_block/private__multi_head_self_attention/dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:­
Yprivate__transformer_block/private__multi_head_self_attention/dense_1/Tensordot/transpose	Transpose-private__token_and_position_embedding/add:z:0_private__transformer_block/private__multi_head_self_attention/dense_1/Tensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ü
Wprivate__transformer_block/private__multi_head_self_attention/dense_1/Tensordot/ReshapeReshape]private__transformer_block/private__multi_head_self_attention/dense_1/Tensordot/transpose:y:0^private__transformer_block/private__multi_head_self_attention/dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÜ
Vprivate__transformer_block/private__multi_head_self_attention/dense_1/Tensordot/MatMulMatMul`private__transformer_block/private__multi_head_self_attention/dense_1/Tensordot/Reshape:output:0fprivate__transformer_block/private__multi_head_self_attention/dense_1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¡
Wprivate__transformer_block/private__multi_head_self_attention/dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 
]private__transformer_block/private__multi_head_self_attention/dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ¿
Xprivate__transformer_block/private__multi_head_self_attention/dense_1/Tensordot/concat_1ConcatV2aprivate__transformer_block/private__multi_head_self_attention/dense_1/Tensordot/GatherV2:output:0`private__transformer_block/private__multi_head_self_attention/dense_1/Tensordot/Const_2:output:0fprivate__transformer_block/private__multi_head_self_attention/dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:Ö
Oprivate__transformer_block/private__multi_head_self_attention/dense_1/TensordotReshape`private__transformer_block/private__multi_head_self_attention/dense_1/Tensordot/MatMul:product:0aprivate__transformer_block/private__multi_head_self_attention/dense_1/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ þ
\private__transformer_block/private__multi_head_self_attention/dense_1/BiasAdd/ReadVariableOpReadVariableOpeprivate__transformer_block_private__multi_head_self_attention_dense_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Ï
Mprivate__transformer_block/private__multi_head_self_attention/dense_1/BiasAddBiasAddXprivate__transformer_block/private__multi_head_self_attention/dense_1/Tensordot:output:0dprivate__transformer_block/private__multi_head_self_attention/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
^private__transformer_block/private__multi_head_self_attention/dense_2/Tensordot/ReadVariableOpReadVariableOpgprivate__transformer_block_private__multi_head_self_attention_dense_2_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype0
Tprivate__transformer_block/private__multi_head_self_attention/dense_2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:¥
Tprivate__transformer_block/private__multi_head_self_attention/dense_2/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       ²
Uprivate__transformer_block/private__multi_head_self_attention/dense_2/Tensordot/ShapeShape-private__token_and_position_embedding/add:z:0*
T0*
_output_shapes
:
]private__transformer_block/private__multi_head_self_attention/dense_2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : Ó
Xprivate__transformer_block/private__multi_head_self_attention/dense_2/Tensordot/GatherV2GatherV2^private__transformer_block/private__multi_head_self_attention/dense_2/Tensordot/Shape:output:0]private__transformer_block/private__multi_head_self_attention/dense_2/Tensordot/free:output:0fprivate__transformer_block/private__multi_head_self_attention/dense_2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:¡
_private__transformer_block/private__multi_head_self_attention/dense_2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ×
Zprivate__transformer_block/private__multi_head_self_attention/dense_2/Tensordot/GatherV2_1GatherV2^private__transformer_block/private__multi_head_self_attention/dense_2/Tensordot/Shape:output:0]private__transformer_block/private__multi_head_self_attention/dense_2/Tensordot/axes:output:0hprivate__transformer_block/private__multi_head_self_attention/dense_2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:
Uprivate__transformer_block/private__multi_head_self_attention/dense_2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: À
Tprivate__transformer_block/private__multi_head_self_attention/dense_2/Tensordot/ProdProdaprivate__transformer_block/private__multi_head_self_attention/dense_2/Tensordot/GatherV2:output:0^private__transformer_block/private__multi_head_self_attention/dense_2/Tensordot/Const:output:0*
T0*
_output_shapes
: ¡
Wprivate__transformer_block/private__multi_head_self_attention/dense_2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: Æ
Vprivate__transformer_block/private__multi_head_self_attention/dense_2/Tensordot/Prod_1Prodcprivate__transformer_block/private__multi_head_self_attention/dense_2/Tensordot/GatherV2_1:output:0`private__transformer_block/private__multi_head_self_attention/dense_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 
[private__transformer_block/private__multi_head_self_attention/dense_2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ´
Vprivate__transformer_block/private__multi_head_self_attention/dense_2/Tensordot/concatConcatV2]private__transformer_block/private__multi_head_self_attention/dense_2/Tensordot/free:output:0]private__transformer_block/private__multi_head_self_attention/dense_2/Tensordot/axes:output:0dprivate__transformer_block/private__multi_head_self_attention/dense_2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:Ë
Uprivate__transformer_block/private__multi_head_self_attention/dense_2/Tensordot/stackPack]private__transformer_block/private__multi_head_self_attention/dense_2/Tensordot/Prod:output:0_private__transformer_block/private__multi_head_self_attention/dense_2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:­
Yprivate__transformer_block/private__multi_head_self_attention/dense_2/Tensordot/transpose	Transpose-private__token_and_position_embedding/add:z:0_private__transformer_block/private__multi_head_self_attention/dense_2/Tensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ü
Wprivate__transformer_block/private__multi_head_self_attention/dense_2/Tensordot/ReshapeReshape]private__transformer_block/private__multi_head_self_attention/dense_2/Tensordot/transpose:y:0^private__transformer_block/private__multi_head_self_attention/dense_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÜ
Vprivate__transformer_block/private__multi_head_self_attention/dense_2/Tensordot/MatMulMatMul`private__transformer_block/private__multi_head_self_attention/dense_2/Tensordot/Reshape:output:0fprivate__transformer_block/private__multi_head_self_attention/dense_2/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¡
Wprivate__transformer_block/private__multi_head_self_attention/dense_2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 
]private__transformer_block/private__multi_head_self_attention/dense_2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ¿
Xprivate__transformer_block/private__multi_head_self_attention/dense_2/Tensordot/concat_1ConcatV2aprivate__transformer_block/private__multi_head_self_attention/dense_2/Tensordot/GatherV2:output:0`private__transformer_block/private__multi_head_self_attention/dense_2/Tensordot/Const_2:output:0fprivate__transformer_block/private__multi_head_self_attention/dense_2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:Ö
Oprivate__transformer_block/private__multi_head_self_attention/dense_2/TensordotReshape`private__transformer_block/private__multi_head_self_attention/dense_2/Tensordot/MatMul:product:0aprivate__transformer_block/private__multi_head_self_attention/dense_2/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ þ
\private__transformer_block/private__multi_head_self_attention/dense_2/BiasAdd/ReadVariableOpReadVariableOpeprivate__transformer_block_private__multi_head_self_attention_dense_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Ï
Mprivate__transformer_block/private__multi_head_self_attention/dense_2/BiasAddBiasAddXprivate__transformer_block/private__multi_head_self_attention/dense_2/Tensordot:output:0dprivate__transformer_block/private__multi_head_self_attention/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
Mprivate__transformer_block/private__multi_head_self_attention/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
Mprivate__transformer_block/private__multi_head_self_attention/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :
Mprivate__transformer_block/private__multi_head_self_attention/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :ß
Kprivate__transformer_block/private__multi_head_self_attention/Reshape/shapePackTprivate__transformer_block/private__multi_head_self_attention/strided_slice:output:0Vprivate__transformer_block/private__multi_head_self_attention/Reshape/shape/1:output:0Vprivate__transformer_block/private__multi_head_self_attention/Reshape/shape/2:output:0Vprivate__transformer_block/private__multi_head_self_attention/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:¿
Eprivate__transformer_block/private__multi_head_self_attention/ReshapeReshapeTprivate__transformer_block/private__multi_head_self_attention/dense/BiasAdd:output:0Tprivate__transformer_block/private__multi_head_self_attention/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¥
Lprivate__transformer_block/private__multi_head_self_attention/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             ¾
Gprivate__transformer_block/private__multi_head_self_attention/transpose	TransposeNprivate__transformer_block/private__multi_head_self_attention/Reshape:output:0Uprivate__transformer_block/private__multi_head_self_attention/transpose/perm:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Oprivate__transformer_block/private__multi_head_self_attention/Reshape_1/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
Oprivate__transformer_block/private__multi_head_self_attention/Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :
Oprivate__transformer_block/private__multi_head_self_attention/Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :ç
Mprivate__transformer_block/private__multi_head_self_attention/Reshape_1/shapePackTprivate__transformer_block/private__multi_head_self_attention/strided_slice:output:0Xprivate__transformer_block/private__multi_head_self_attention/Reshape_1/shape/1:output:0Xprivate__transformer_block/private__multi_head_self_attention/Reshape_1/shape/2:output:0Xprivate__transformer_block/private__multi_head_self_attention/Reshape_1/shape/3:output:0*
N*
T0*
_output_shapes
:Å
Gprivate__transformer_block/private__multi_head_self_attention/Reshape_1ReshapeVprivate__transformer_block/private__multi_head_self_attention/dense_1/BiasAdd:output:0Vprivate__transformer_block/private__multi_head_self_attention/Reshape_1/shape:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ§
Nprivate__transformer_block/private__multi_head_self_attention/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             Ä
Iprivate__transformer_block/private__multi_head_self_attention/transpose_1	TransposePprivate__transformer_block/private__multi_head_self_attention/Reshape_1:output:0Wprivate__transformer_block/private__multi_head_self_attention/transpose_1/perm:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Oprivate__transformer_block/private__multi_head_self_attention/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
Oprivate__transformer_block/private__multi_head_self_attention/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :
Oprivate__transformer_block/private__multi_head_self_attention/Reshape_2/shape/3Const*
_output_shapes
: *
dtype0*
value	B :ç
Mprivate__transformer_block/private__multi_head_self_attention/Reshape_2/shapePackTprivate__transformer_block/private__multi_head_self_attention/strided_slice:output:0Xprivate__transformer_block/private__multi_head_self_attention/Reshape_2/shape/1:output:0Xprivate__transformer_block/private__multi_head_self_attention/Reshape_2/shape/2:output:0Xprivate__transformer_block/private__multi_head_self_attention/Reshape_2/shape/3:output:0*
N*
T0*
_output_shapes
:Å
Gprivate__transformer_block/private__multi_head_self_attention/Reshape_2ReshapeVprivate__transformer_block/private__multi_head_self_attention/dense_2/BiasAdd:output:0Vprivate__transformer_block/private__multi_head_self_attention/Reshape_2/shape:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ§
Nprivate__transformer_block/private__multi_head_self_attention/transpose_2/permConst*
_output_shapes
:*
dtype0*%
valueB"             Ä
Iprivate__transformer_block/private__multi_head_self_attention/transpose_2	TransposePprivate__transformer_block/private__multi_head_self_attention/Reshape_2:output:0Wprivate__transformer_block/private__multi_head_self_attention/transpose_2/perm:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÊ
Dprivate__transformer_block/private__multi_head_self_attention/MatMulBatchMatMulV2Kprivate__transformer_block/private__multi_head_self_attention/transpose:y:0Mprivate__transformer_block/private__multi_head_self_attention/transpose_1:y:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
adj_y(Â
Eprivate__transformer_block/private__multi_head_self_attention/Shape_1ShapeMprivate__transformer_block/private__multi_head_self_attention/transpose_1:y:0*
T0*
_output_shapes
:¦
Sprivate__transformer_block/private__multi_head_self_attention/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ
Uprivate__transformer_block/private__multi_head_self_attention/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
Uprivate__transformer_block/private__multi_head_self_attention/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
Mprivate__transformer_block/private__multi_head_self_attention/strided_slice_1StridedSliceNprivate__transformer_block/private__multi_head_self_attention/Shape_1:output:0\private__transformer_block/private__multi_head_self_attention/strided_slice_1/stack:output:0^private__transformer_block/private__multi_head_self_attention/strided_slice_1/stack_1:output:0^private__transformer_block/private__multi_head_self_attention/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskÒ
Bprivate__transformer_block/private__multi_head_self_attention/CastCastVprivate__transformer_block/private__multi_head_self_attention/strided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: ³
Bprivate__transformer_block/private__multi_head_self_attention/SqrtSqrtFprivate__transformer_block/private__multi_head_self_attention/Cast:y:0*
T0*
_output_shapes
: ³
Eprivate__transformer_block/private__multi_head_self_attention/truedivRealDivMprivate__transformer_block/private__multi_head_self_attention/MatMul:output:0Fprivate__transformer_block/private__multi_head_self_attention/Sqrt:y:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿç
Eprivate__transformer_block/private__multi_head_self_attention/SoftmaxSoftmaxIprivate__transformer_block/private__multi_head_self_attention/truediv:z:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿº
Fprivate__transformer_block/private__multi_head_self_attention/MatMul_1BatchMatMulV2Oprivate__transformer_block/private__multi_head_self_attention/Softmax:softmax:0Mprivate__transformer_block/private__multi_head_self_attention/transpose_2:y:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ§
Nprivate__transformer_block/private__multi_head_self_attention/transpose_3/permConst*
_output_shapes
:*
dtype0*%
valueB"             Ã
Iprivate__transformer_block/private__multi_head_self_attention/transpose_3	TransposeOprivate__transformer_block/private__multi_head_self_attention/MatMul_1:output:0Wprivate__transformer_block/private__multi_head_self_attention/transpose_3/perm:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Oprivate__transformer_block/private__multi_head_self_attention/Reshape_3/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
Oprivate__transformer_block/private__multi_head_self_attention/Reshape_3/shape/2Const*
_output_shapes
: *
dtype0*
value	B : 
Mprivate__transformer_block/private__multi_head_self_attention/Reshape_3/shapePackTprivate__transformer_block/private__multi_head_self_attention/strided_slice:output:0Xprivate__transformer_block/private__multi_head_self_attention/Reshape_3/shape/1:output:0Xprivate__transformer_block/private__multi_head_self_attention/Reshape_3/shape/2:output:0*
N*
T0*
_output_shapes
:¸
Gprivate__transformer_block/private__multi_head_self_attention/Reshape_3ReshapeMprivate__transformer_block/private__multi_head_self_attention/transpose_3:y:0Vprivate__transformer_block/private__multi_head_self_attention/Reshape_3/shape:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
^private__transformer_block/private__multi_head_self_attention/dense_3/Tensordot/ReadVariableOpReadVariableOpgprivate__transformer_block_private__multi_head_self_attention_dense_3_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype0
Tprivate__transformer_block/private__multi_head_self_attention/dense_3/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:¥
Tprivate__transformer_block/private__multi_head_self_attention/dense_3/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       Õ
Uprivate__transformer_block/private__multi_head_self_attention/dense_3/Tensordot/ShapeShapePprivate__transformer_block/private__multi_head_self_attention/Reshape_3:output:0*
T0*
_output_shapes
:
]private__transformer_block/private__multi_head_self_attention/dense_3/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : Ó
Xprivate__transformer_block/private__multi_head_self_attention/dense_3/Tensordot/GatherV2GatherV2^private__transformer_block/private__multi_head_self_attention/dense_3/Tensordot/Shape:output:0]private__transformer_block/private__multi_head_self_attention/dense_3/Tensordot/free:output:0fprivate__transformer_block/private__multi_head_self_attention/dense_3/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:¡
_private__transformer_block/private__multi_head_self_attention/dense_3/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ×
Zprivate__transformer_block/private__multi_head_self_attention/dense_3/Tensordot/GatherV2_1GatherV2^private__transformer_block/private__multi_head_self_attention/dense_3/Tensordot/Shape:output:0]private__transformer_block/private__multi_head_self_attention/dense_3/Tensordot/axes:output:0hprivate__transformer_block/private__multi_head_self_attention/dense_3/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:
Uprivate__transformer_block/private__multi_head_self_attention/dense_3/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: À
Tprivate__transformer_block/private__multi_head_self_attention/dense_3/Tensordot/ProdProdaprivate__transformer_block/private__multi_head_self_attention/dense_3/Tensordot/GatherV2:output:0^private__transformer_block/private__multi_head_self_attention/dense_3/Tensordot/Const:output:0*
T0*
_output_shapes
: ¡
Wprivate__transformer_block/private__multi_head_self_attention/dense_3/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: Æ
Vprivate__transformer_block/private__multi_head_self_attention/dense_3/Tensordot/Prod_1Prodcprivate__transformer_block/private__multi_head_self_attention/dense_3/Tensordot/GatherV2_1:output:0`private__transformer_block/private__multi_head_self_attention/dense_3/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 
[private__transformer_block/private__multi_head_self_attention/dense_3/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ´
Vprivate__transformer_block/private__multi_head_self_attention/dense_3/Tensordot/concatConcatV2]private__transformer_block/private__multi_head_self_attention/dense_3/Tensordot/free:output:0]private__transformer_block/private__multi_head_self_attention/dense_3/Tensordot/axes:output:0dprivate__transformer_block/private__multi_head_self_attention/dense_3/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:Ë
Uprivate__transformer_block/private__multi_head_self_attention/dense_3/Tensordot/stackPack]private__transformer_block/private__multi_head_self_attention/dense_3/Tensordot/Prod:output:0_private__transformer_block/private__multi_head_self_attention/dense_3/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:Ø
Yprivate__transformer_block/private__multi_head_self_attention/dense_3/Tensordot/transpose	TransposePprivate__transformer_block/private__multi_head_self_attention/Reshape_3:output:0_private__transformer_block/private__multi_head_self_attention/dense_3/Tensordot/concat:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ Ü
Wprivate__transformer_block/private__multi_head_self_attention/dense_3/Tensordot/ReshapeReshape]private__transformer_block/private__multi_head_self_attention/dense_3/Tensordot/transpose:y:0^private__transformer_block/private__multi_head_self_attention/dense_3/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÜ
Vprivate__transformer_block/private__multi_head_self_attention/dense_3/Tensordot/MatMulMatMul`private__transformer_block/private__multi_head_self_attention/dense_3/Tensordot/Reshape:output:0fprivate__transformer_block/private__multi_head_self_attention/dense_3/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¡
Wprivate__transformer_block/private__multi_head_self_attention/dense_3/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 
]private__transformer_block/private__multi_head_self_attention/dense_3/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ¿
Xprivate__transformer_block/private__multi_head_self_attention/dense_3/Tensordot/concat_1ConcatV2aprivate__transformer_block/private__multi_head_self_attention/dense_3/Tensordot/GatherV2:output:0`private__transformer_block/private__multi_head_self_attention/dense_3/Tensordot/Const_2:output:0fprivate__transformer_block/private__multi_head_self_attention/dense_3/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:Þ
Oprivate__transformer_block/private__multi_head_self_attention/dense_3/TensordotReshape`private__transformer_block/private__multi_head_self_attention/dense_3/Tensordot/MatMul:product:0aprivate__transformer_block/private__multi_head_self_attention/dense_3/Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ þ
\private__transformer_block/private__multi_head_self_attention/dense_3/BiasAdd/ReadVariableOpReadVariableOpeprivate__transformer_block_private__multi_head_self_attention_dense_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0×
Mprivate__transformer_block/private__multi_head_self_attention/dense_3/BiasAddBiasAddXprivate__transformer_block/private__multi_head_self_attention/dense_3/Tensordot:output:0dprivate__transformer_block/private__multi_head_self_attention/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ u
0private__transformer_block/dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?
.private__transformer_block/dropout/dropout/MulMulVprivate__transformer_block/private__multi_head_self_attention/dense_3/BiasAdd:output:09private__transformer_block/dropout/dropout/Const:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ¶
0private__transformer_block/dropout/dropout/ShapeShapeVprivate__transformer_block/private__multi_head_self_attention/dense_3/BiasAdd:output:0*
T0*
_output_shapes
:ß
Gprivate__transformer_block/dropout/dropout/random_uniform/RandomUniformRandomUniform9private__transformer_block/dropout/dropout/Shape:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *
dtype0~
9private__transformer_block/dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=
7private__transformer_block/dropout/dropout/GreaterEqualGreaterEqualPprivate__transformer_block/dropout/dropout/random_uniform/RandomUniform:output:0Bprivate__transformer_block/dropout/dropout/GreaterEqual/y:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ Â
/private__transformer_block/dropout/dropout/CastCast;private__transformer_block/dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ß
0private__transformer_block/dropout/dropout/Mul_1Mul2private__transformer_block/dropout/dropout/Mul:z:03private__transformer_block/dropout/dropout/Cast:y:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ Ã
private__transformer_block/addAddV2-private__token_and_position_embedding/add:z:04private__transformer_block/dropout/dropout/Mul_1:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
Mprivate__transformer_block/layer_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
;private__transformer_block/layer_normalization/moments/meanMean"private__transformer_block/add:z:0Vprivate__transformer_block/layer_normalization/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(Ð
Cprivate__transformer_block/layer_normalization/moments/StopGradientStopGradientDprivate__transformer_block/layer_normalization/moments/mean:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Hprivate__transformer_block/layer_normalization/moments/SquaredDifferenceSquaredDifference"private__transformer_block/add:z:0Lprivate__transformer_block/layer_normalization/moments/StopGradient:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
Qprivate__transformer_block/layer_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:¹
?private__transformer_block/layer_normalization/moments/varianceMeanLprivate__transformer_block/layer_normalization/moments/SquaredDifference:z:0Zprivate__transformer_block/layer_normalization/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(
>private__transformer_block/layer_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½75
<private__transformer_block/layer_normalization/batchnorm/addAddV2Hprivate__transformer_block/layer_normalization/moments/variance:output:0Gprivate__transformer_block/layer_normalization/batchnorm/add/y:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
>private__transformer_block/layer_normalization/batchnorm/RsqrtRsqrt@private__transformer_block/layer_normalization/batchnorm/add:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ
Kprivate__transformer_block/layer_normalization/batchnorm/mul/ReadVariableOpReadVariableOpTprivate__transformer_block_layer_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype0
<private__transformer_block/layer_normalization/batchnorm/mulMulBprivate__transformer_block/layer_normalization/batchnorm/Rsqrt:y:0Sprivate__transformer_block/layer_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ â
>private__transformer_block/layer_normalization/batchnorm/mul_1Mul"private__transformer_block/add:z:0@private__transformer_block/layer_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
>private__transformer_block/layer_normalization/batchnorm/mul_2MulDprivate__transformer_block/layer_normalization/moments/mean:output:0@private__transformer_block/layer_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ô
Gprivate__transformer_block/layer_normalization/batchnorm/ReadVariableOpReadVariableOpPprivate__transformer_block_layer_normalization_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype0
<private__transformer_block/layer_normalization/batchnorm/subSubOprivate__transformer_block/layer_normalization/batchnorm/ReadVariableOp:value:0Bprivate__transformer_block/layer_normalization/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
>private__transformer_block/layer_normalization/batchnorm/add_1AddV2Bprivate__transformer_block/layer_normalization/batchnorm/mul_1:z:0@private__transformer_block/layer_normalization/batchnorm/sub:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ö
Fprivate__transformer_block/sequential/dense_4/Tensordot/ReadVariableOpReadVariableOpOprivate__transformer_block_sequential_dense_4_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype0
<private__transformer_block/sequential/dense_4/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:
<private__transformer_block/sequential/dense_4/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       ¯
=private__transformer_block/sequential/dense_4/Tensordot/ShapeShapeBprivate__transformer_block/layer_normalization/batchnorm/add_1:z:0*
T0*
_output_shapes
:
Eprivate__transformer_block/sequential/dense_4/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ó
@private__transformer_block/sequential/dense_4/Tensordot/GatherV2GatherV2Fprivate__transformer_block/sequential/dense_4/Tensordot/Shape:output:0Eprivate__transformer_block/sequential/dense_4/Tensordot/free:output:0Nprivate__transformer_block/sequential/dense_4/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:
Gprivate__transformer_block/sequential/dense_4/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ÷
Bprivate__transformer_block/sequential/dense_4/Tensordot/GatherV2_1GatherV2Fprivate__transformer_block/sequential/dense_4/Tensordot/Shape:output:0Eprivate__transformer_block/sequential/dense_4/Tensordot/axes:output:0Pprivate__transformer_block/sequential/dense_4/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:
=private__transformer_block/sequential/dense_4/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ø
<private__transformer_block/sequential/dense_4/Tensordot/ProdProdIprivate__transformer_block/sequential/dense_4/Tensordot/GatherV2:output:0Fprivate__transformer_block/sequential/dense_4/Tensordot/Const:output:0*
T0*
_output_shapes
: 
?private__transformer_block/sequential/dense_4/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: þ
>private__transformer_block/sequential/dense_4/Tensordot/Prod_1ProdKprivate__transformer_block/sequential/dense_4/Tensordot/GatherV2_1:output:0Hprivate__transformer_block/sequential/dense_4/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 
Cprivate__transformer_block/sequential/dense_4/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ô
>private__transformer_block/sequential/dense_4/Tensordot/concatConcatV2Eprivate__transformer_block/sequential/dense_4/Tensordot/free:output:0Eprivate__transformer_block/sequential/dense_4/Tensordot/axes:output:0Lprivate__transformer_block/sequential/dense_4/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:
=private__transformer_block/sequential/dense_4/Tensordot/stackPackEprivate__transformer_block/sequential/dense_4/Tensordot/Prod:output:0Gprivate__transformer_block/sequential/dense_4/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:
Aprivate__transformer_block/sequential/dense_4/Tensordot/transpose	TransposeBprivate__transformer_block/layer_normalization/batchnorm/add_1:z:0Gprivate__transformer_block/sequential/dense_4/Tensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
?private__transformer_block/sequential/dense_4/Tensordot/ReshapeReshapeEprivate__transformer_block/sequential/dense_4/Tensordot/transpose:y:0Fprivate__transformer_block/sequential/dense_4/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
>private__transformer_block/sequential/dense_4/Tensordot/MatMulMatMulHprivate__transformer_block/sequential/dense_4/Tensordot/Reshape:output:0Nprivate__transformer_block/sequential/dense_4/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
?private__transformer_block/sequential/dense_4/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 
Eprivate__transformer_block/sequential/dense_4/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ß
@private__transformer_block/sequential/dense_4/Tensordot/concat_1ConcatV2Iprivate__transformer_block/sequential/dense_4/Tensordot/GatherV2:output:0Hprivate__transformer_block/sequential/dense_4/Tensordot/Const_2:output:0Nprivate__transformer_block/sequential/dense_4/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
7private__transformer_block/sequential/dense_4/TensordotReshapeHprivate__transformer_block/sequential/dense_4/Tensordot/MatMul:product:0Iprivate__transformer_block/sequential/dense_4/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Î
Dprivate__transformer_block/sequential/dense_4/BiasAdd/ReadVariableOpReadVariableOpMprivate__transformer_block_sequential_dense_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
5private__transformer_block/sequential/dense_4/BiasAddBiasAdd@private__transformer_block/sequential/dense_4/Tensordot:output:0Lprivate__transformer_block/sequential/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ±
2private__transformer_block/sequential/dense_4/ReluRelu>private__transformer_block/sequential/dense_4/BiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ö
Fprivate__transformer_block/sequential/dense_5/Tensordot/ReadVariableOpReadVariableOpOprivate__transformer_block_sequential_dense_5_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype0
<private__transformer_block/sequential/dense_5/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:
<private__transformer_block/sequential/dense_5/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       ­
=private__transformer_block/sequential/dense_5/Tensordot/ShapeShape@private__transformer_block/sequential/dense_4/Relu:activations:0*
T0*
_output_shapes
:
Eprivate__transformer_block/sequential/dense_5/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ó
@private__transformer_block/sequential/dense_5/Tensordot/GatherV2GatherV2Fprivate__transformer_block/sequential/dense_5/Tensordot/Shape:output:0Eprivate__transformer_block/sequential/dense_5/Tensordot/free:output:0Nprivate__transformer_block/sequential/dense_5/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:
Gprivate__transformer_block/sequential/dense_5/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ÷
Bprivate__transformer_block/sequential/dense_5/Tensordot/GatherV2_1GatherV2Fprivate__transformer_block/sequential/dense_5/Tensordot/Shape:output:0Eprivate__transformer_block/sequential/dense_5/Tensordot/axes:output:0Pprivate__transformer_block/sequential/dense_5/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:
=private__transformer_block/sequential/dense_5/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ø
<private__transformer_block/sequential/dense_5/Tensordot/ProdProdIprivate__transformer_block/sequential/dense_5/Tensordot/GatherV2:output:0Fprivate__transformer_block/sequential/dense_5/Tensordot/Const:output:0*
T0*
_output_shapes
: 
?private__transformer_block/sequential/dense_5/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: þ
>private__transformer_block/sequential/dense_5/Tensordot/Prod_1ProdKprivate__transformer_block/sequential/dense_5/Tensordot/GatherV2_1:output:0Hprivate__transformer_block/sequential/dense_5/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 
Cprivate__transformer_block/sequential/dense_5/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ô
>private__transformer_block/sequential/dense_5/Tensordot/concatConcatV2Eprivate__transformer_block/sequential/dense_5/Tensordot/free:output:0Eprivate__transformer_block/sequential/dense_5/Tensordot/axes:output:0Lprivate__transformer_block/sequential/dense_5/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:
=private__transformer_block/sequential/dense_5/Tensordot/stackPackEprivate__transformer_block/sequential/dense_5/Tensordot/Prod:output:0Gprivate__transformer_block/sequential/dense_5/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:
Aprivate__transformer_block/sequential/dense_5/Tensordot/transpose	Transpose@private__transformer_block/sequential/dense_4/Relu:activations:0Gprivate__transformer_block/sequential/dense_5/Tensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
?private__transformer_block/sequential/dense_5/Tensordot/ReshapeReshapeEprivate__transformer_block/sequential/dense_5/Tensordot/transpose:y:0Fprivate__transformer_block/sequential/dense_5/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
>private__transformer_block/sequential/dense_5/Tensordot/MatMulMatMulHprivate__transformer_block/sequential/dense_5/Tensordot/Reshape:output:0Nprivate__transformer_block/sequential/dense_5/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
?private__transformer_block/sequential/dense_5/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 
Eprivate__transformer_block/sequential/dense_5/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ß
@private__transformer_block/sequential/dense_5/Tensordot/concat_1ConcatV2Iprivate__transformer_block/sequential/dense_5/Tensordot/GatherV2:output:0Hprivate__transformer_block/sequential/dense_5/Tensordot/Const_2:output:0Nprivate__transformer_block/sequential/dense_5/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
7private__transformer_block/sequential/dense_5/TensordotReshapeHprivate__transformer_block/sequential/dense_5/Tensordot/MatMul:product:0Iprivate__transformer_block/sequential/dense_5/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Î
Dprivate__transformer_block/sequential/dense_5/BiasAdd/ReadVariableOpReadVariableOpMprivate__transformer_block_sequential_dense_5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
5private__transformer_block/sequential/dense_5/BiasAddBiasAdd@private__transformer_block/sequential/dense_5/Tensordot:output:0Lprivate__transformer_block/sequential/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ w
2private__transformer_block/dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?ë
0private__transformer_block/dropout_1/dropout/MulMul>private__transformer_block/sequential/dense_5/BiasAdd:output:0;private__transformer_block/dropout_1/dropout/Const:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
2private__transformer_block/dropout_1/dropout/ShapeShape>private__transformer_block/sequential/dense_5/BiasAdd:output:0*
T0*
_output_shapes
:Û
Iprivate__transformer_block/dropout_1/dropout/random_uniform/RandomUniformRandomUniform;private__transformer_block/dropout_1/dropout/Shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
dtype0
;private__transformer_block/dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=
9private__transformer_block/dropout_1/dropout/GreaterEqualGreaterEqualRprivate__transformer_block/dropout_1/dropout/random_uniform/RandomUniform:output:0Dprivate__transformer_block/dropout_1/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¾
1private__transformer_block/dropout_1/dropout/CastCast=private__transformer_block/dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ý
2private__transformer_block/dropout_1/dropout/Mul_1Mul4private__transformer_block/dropout_1/dropout/Mul:z:05private__transformer_block/dropout_1/dropout/Cast:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ü
 private__transformer_block/add_1AddV2Bprivate__transformer_block/layer_normalization/batchnorm/add_1:z:06private__transformer_block/dropout_1/dropout/Mul_1:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
Oprivate__transformer_block/layer_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
=private__transformer_block/layer_normalization_1/moments/meanMean$private__transformer_block/add_1:z:0Xprivate__transformer_block/layer_normalization_1/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(Ô
Eprivate__transformer_block/layer_normalization_1/moments/StopGradientStopGradientFprivate__transformer_block/layer_normalization_1/moments/mean:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Jprivate__transformer_block/layer_normalization_1/moments/SquaredDifferenceSquaredDifference$private__transformer_block/add_1:z:0Nprivate__transformer_block/layer_normalization_1/moments/StopGradient:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
Sprivate__transformer_block/layer_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:¿
Aprivate__transformer_block/layer_normalization_1/moments/varianceMeanNprivate__transformer_block/layer_normalization_1/moments/SquaredDifference:z:0\private__transformer_block/layer_normalization_1/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(
@private__transformer_block/layer_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½75
>private__transformer_block/layer_normalization_1/batchnorm/addAddV2Jprivate__transformer_block/layer_normalization_1/moments/variance:output:0Iprivate__transformer_block/layer_normalization_1/batchnorm/add/y:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄ
@private__transformer_block/layer_normalization_1/batchnorm/RsqrtRsqrtBprivate__transformer_block/layer_normalization_1/batchnorm/add:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿà
Mprivate__transformer_block/layer_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOpVprivate__transformer_block_layer_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype0
>private__transformer_block/layer_normalization_1/batchnorm/mulMulDprivate__transformer_block/layer_normalization_1/batchnorm/Rsqrt:y:0Uprivate__transformer_block/layer_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ è
@private__transformer_block/layer_normalization_1/batchnorm/mul_1Mul$private__transformer_block/add_1:z:0Bprivate__transformer_block/layer_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
@private__transformer_block/layer_normalization_1/batchnorm/mul_2MulFprivate__transformer_block/layer_normalization_1/moments/mean:output:0Bprivate__transformer_block/layer_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ø
Iprivate__transformer_block/layer_normalization_1/batchnorm/ReadVariableOpReadVariableOpRprivate__transformer_block_layer_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype0
>private__transformer_block/layer_normalization_1/batchnorm/subSubQprivate__transformer_block/layer_normalization_1/batchnorm/ReadVariableOp:value:0Dprivate__transformer_block/layer_normalization_1/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
@private__transformer_block/layer_normalization_1/batchnorm/add_1AddV2Dprivate__transformer_block/layer_normalization_1/batchnorm/mul_1:z:0Bprivate__transformer_block/layer_normalization_1/batchnorm/sub:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ q
/global_average_pooling1d/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :×
global_average_pooling1d/MeanMeanDprivate__transformer_block/layer_normalization_1/batchnorm/add_1:z:08global_average_pooling1d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ \
dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?
dropout_2/dropout/MulMul&global_average_pooling1d/Mean:output:0 dropout_2/dropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ m
dropout_2/dropout/ShapeShape&global_average_pooling1d/Mean:output:0*
T0*
_output_shapes
: 
.dropout_2/dropout/random_uniform/RandomUniformRandomUniform dropout_2/dropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
dtype0e
 dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=Ä
dropout_2/dropout/GreaterEqualGreaterEqual7dropout_2/dropout/random_uniform/RandomUniform:output:0)dropout_2/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
dropout_2/dropout/CastCast"dropout_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
dropout_2/dropout/Mul_1Muldropout_2/dropout/Mul:z:0dropout_2/dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource*
_output_shapes

: *
dtype0
dense_6/MatMulMatMuldropout_2/dropout/Mul_1:z:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
dense_6/ReluReludense_6/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\
dropout_3/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?
dropout_3/dropout/MulMuldense_6/Relu:activations:0 dropout_3/dropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
dropout_3/dropout/ShapeShapedense_6/Relu:activations:0*
T0*
_output_shapes
: 
.dropout_3/dropout/random_uniform/RandomUniformRandomUniform dropout_3/dropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0e
 dropout_3/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=Ä
dropout_3/dropout/GreaterEqualGreaterEqual7dropout_3/dropout/random_uniform/RandomUniform:output:0)dropout_3/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dropout_3/dropout/CastCast"dropout_3/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dropout_3/dropout/Mul_1Muldropout_3/dropout/Mul:z:0dropout_3/dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_7/MatMul/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_7/MatMulMatMuldropout_3/dropout/Mul_1:z:0%dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_7/BiasAddBiasAdddense_7/MatMul:product:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
dense_7/SoftmaxSoftmaxdense_7/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
IdentityIdentitydense_7/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
NoOpNoOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp^dense_7/BiasAdd/ReadVariableOp^dense_7/MatMul/ReadVariableOpA^private__token_and_position_embedding/embedding/embedding_lookupC^private__token_and_position_embedding/embedding_1/embedding_lookupH^private__transformer_block/layer_normalization/batchnorm/ReadVariableOpL^private__transformer_block/layer_normalization/batchnorm/mul/ReadVariableOpJ^private__transformer_block/layer_normalization_1/batchnorm/ReadVariableOpN^private__transformer_block/layer_normalization_1/batchnorm/mul/ReadVariableOp[^private__transformer_block/private__multi_head_self_attention/dense/BiasAdd/ReadVariableOp]^private__transformer_block/private__multi_head_self_attention/dense/Tensordot/ReadVariableOp]^private__transformer_block/private__multi_head_self_attention/dense_1/BiasAdd/ReadVariableOp_^private__transformer_block/private__multi_head_self_attention/dense_1/Tensordot/ReadVariableOp]^private__transformer_block/private__multi_head_self_attention/dense_2/BiasAdd/ReadVariableOp_^private__transformer_block/private__multi_head_self_attention/dense_2/Tensordot/ReadVariableOp]^private__transformer_block/private__multi_head_self_attention/dense_3/BiasAdd/ReadVariableOp_^private__transformer_block/private__multi_head_self_attention/dense_3/Tensordot/ReadVariableOpE^private__transformer_block/sequential/dense_4/BiasAdd/ReadVariableOpG^private__transformer_block/sequential/dense_4/Tensordot/ReadVariableOpE^private__transformer_block/sequential/dense_5/BiasAdd/ReadVariableOpG^private__transformer_block/sequential/dense_5/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : 2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp2@
dense_7/BiasAdd/ReadVariableOpdense_7/BiasAdd/ReadVariableOp2>
dense_7/MatMul/ReadVariableOpdense_7/MatMul/ReadVariableOp2
@private__token_and_position_embedding/embedding/embedding_lookup@private__token_and_position_embedding/embedding/embedding_lookup2
Bprivate__token_and_position_embedding/embedding_1/embedding_lookupBprivate__token_and_position_embedding/embedding_1/embedding_lookup2
Gprivate__transformer_block/layer_normalization/batchnorm/ReadVariableOpGprivate__transformer_block/layer_normalization/batchnorm/ReadVariableOp2
Kprivate__transformer_block/layer_normalization/batchnorm/mul/ReadVariableOpKprivate__transformer_block/layer_normalization/batchnorm/mul/ReadVariableOp2
Iprivate__transformer_block/layer_normalization_1/batchnorm/ReadVariableOpIprivate__transformer_block/layer_normalization_1/batchnorm/ReadVariableOp2
Mprivate__transformer_block/layer_normalization_1/batchnorm/mul/ReadVariableOpMprivate__transformer_block/layer_normalization_1/batchnorm/mul/ReadVariableOp2¸
Zprivate__transformer_block/private__multi_head_self_attention/dense/BiasAdd/ReadVariableOpZprivate__transformer_block/private__multi_head_self_attention/dense/BiasAdd/ReadVariableOp2¼
\private__transformer_block/private__multi_head_self_attention/dense/Tensordot/ReadVariableOp\private__transformer_block/private__multi_head_self_attention/dense/Tensordot/ReadVariableOp2¼
\private__transformer_block/private__multi_head_self_attention/dense_1/BiasAdd/ReadVariableOp\private__transformer_block/private__multi_head_self_attention/dense_1/BiasAdd/ReadVariableOp2À
^private__transformer_block/private__multi_head_self_attention/dense_1/Tensordot/ReadVariableOp^private__transformer_block/private__multi_head_self_attention/dense_1/Tensordot/ReadVariableOp2¼
\private__transformer_block/private__multi_head_self_attention/dense_2/BiasAdd/ReadVariableOp\private__transformer_block/private__multi_head_self_attention/dense_2/BiasAdd/ReadVariableOp2À
^private__transformer_block/private__multi_head_self_attention/dense_2/Tensordot/ReadVariableOp^private__transformer_block/private__multi_head_self_attention/dense_2/Tensordot/ReadVariableOp2¼
\private__transformer_block/private__multi_head_self_attention/dense_3/BiasAdd/ReadVariableOp\private__transformer_block/private__multi_head_self_attention/dense_3/BiasAdd/ReadVariableOp2À
^private__transformer_block/private__multi_head_self_attention/dense_3/Tensordot/ReadVariableOp^private__transformer_block/private__multi_head_self_attention/dense_3/Tensordot/ReadVariableOp2
Dprivate__transformer_block/sequential/dense_4/BiasAdd/ReadVariableOpDprivate__transformer_block/sequential/dense_4/BiasAdd/ReadVariableOp2
Fprivate__transformer_block/sequential/dense_4/Tensordot/ReadVariableOpFprivate__transformer_block/sequential/dense_4/Tensordot/ReadVariableOp2
Dprivate__transformer_block/sequential/dense_5/BiasAdd/ReadVariableOpDprivate__transformer_block/sequential/dense_5/BiasAdd/ReadVariableOp2
Fprivate__transformer_block/sequential/dense_5/Tensordot/ReadVariableOpFprivate__transformer_block/sequential/dense_5/Tensordot/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
û
²
D__inference_private__token_and_position_embedding_layer_call_fn_6777
x
unknown:	 
	unknown_0:	Æ 
identity¢StatefulPartitionedCallô
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *h
fcRa
___inference_private__token_and_position_embedding_layer_call_and_return_conditional_losses_4919t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:K G
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_namex

n
R__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_4885

inputs
identityX
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :p
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ^
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
õú
«
T__inference_private__transformer_block_layer_call_and_return_conditional_losses_5169

inputs\
Jprivate__multi_head_self_attention_dense_tensordot_readvariableop_resource:  V
Hprivate__multi_head_self_attention_dense_biasadd_readvariableop_resource: ^
Lprivate__multi_head_self_attention_dense_1_tensordot_readvariableop_resource:  X
Jprivate__multi_head_self_attention_dense_1_biasadd_readvariableop_resource: ^
Lprivate__multi_head_self_attention_dense_2_tensordot_readvariableop_resource:  X
Jprivate__multi_head_self_attention_dense_2_biasadd_readvariableop_resource: ^
Lprivate__multi_head_self_attention_dense_3_tensordot_readvariableop_resource:  X
Jprivate__multi_head_self_attention_dense_3_biasadd_readvariableop_resource: G
9layer_normalization_batchnorm_mul_readvariableop_resource: C
5layer_normalization_batchnorm_readvariableop_resource: F
4sequential_dense_4_tensordot_readvariableop_resource:  @
2sequential_dense_4_biasadd_readvariableop_resource: F
4sequential_dense_5_tensordot_readvariableop_resource:  @
2sequential_dense_5_biasadd_readvariableop_resource: I
;layer_normalization_1_batchnorm_mul_readvariableop_resource: E
7layer_normalization_1_batchnorm_readvariableop_resource: 
identity¢,layer_normalization/batchnorm/ReadVariableOp¢0layer_normalization/batchnorm/mul/ReadVariableOp¢.layer_normalization_1/batchnorm/ReadVariableOp¢2layer_normalization_1/batchnorm/mul/ReadVariableOp¢?private__multi_head_self_attention/dense/BiasAdd/ReadVariableOp¢Aprivate__multi_head_self_attention/dense/Tensordot/ReadVariableOp¢Aprivate__multi_head_self_attention/dense_1/BiasAdd/ReadVariableOp¢Cprivate__multi_head_self_attention/dense_1/Tensordot/ReadVariableOp¢Aprivate__multi_head_self_attention/dense_2/BiasAdd/ReadVariableOp¢Cprivate__multi_head_self_attention/dense_2/Tensordot/ReadVariableOp¢Aprivate__multi_head_self_attention/dense_3/BiasAdd/ReadVariableOp¢Cprivate__multi_head_self_attention/dense_3/Tensordot/ReadVariableOp¢)sequential/dense_4/BiasAdd/ReadVariableOp¢+sequential/dense_4/Tensordot/ReadVariableOp¢)sequential/dense_5/BiasAdd/ReadVariableOp¢+sequential/dense_5/Tensordot/ReadVariableOp^
(private__multi_head_self_attention/ShapeShapeinputs*
T0*
_output_shapes
:
6private__multi_head_self_attention/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
8private__multi_head_self_attention/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
8private__multi_head_self_attention/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
0private__multi_head_self_attention/strided_sliceStridedSlice1private__multi_head_self_attention/Shape:output:0?private__multi_head_self_attention/strided_slice/stack:output:0Aprivate__multi_head_self_attention/strided_slice/stack_1:output:0Aprivate__multi_head_self_attention/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskÌ
Aprivate__multi_head_self_attention/dense/Tensordot/ReadVariableOpReadVariableOpJprivate__multi_head_self_attention_dense_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype0
7private__multi_head_self_attention/dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:
7private__multi_head_self_attention/dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       n
8private__multi_head_self_attention/dense/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:
@private__multi_head_self_attention/dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ß
;private__multi_head_self_attention/dense/Tensordot/GatherV2GatherV2Aprivate__multi_head_self_attention/dense/Tensordot/Shape:output:0@private__multi_head_self_attention/dense/Tensordot/free:output:0Iprivate__multi_head_self_attention/dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:
Bprivate__multi_head_self_attention/dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ã
=private__multi_head_self_attention/dense/Tensordot/GatherV2_1GatherV2Aprivate__multi_head_self_attention/dense/Tensordot/Shape:output:0@private__multi_head_self_attention/dense/Tensordot/axes:output:0Kprivate__multi_head_self_attention/dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:
8private__multi_head_self_attention/dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: é
7private__multi_head_self_attention/dense/Tensordot/ProdProdDprivate__multi_head_self_attention/dense/Tensordot/GatherV2:output:0Aprivate__multi_head_self_attention/dense/Tensordot/Const:output:0*
T0*
_output_shapes
: 
:private__multi_head_self_attention/dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ï
9private__multi_head_self_attention/dense/Tensordot/Prod_1ProdFprivate__multi_head_self_attention/dense/Tensordot/GatherV2_1:output:0Cprivate__multi_head_self_attention/dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 
>private__multi_head_self_attention/dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : À
9private__multi_head_self_attention/dense/Tensordot/concatConcatV2@private__multi_head_self_attention/dense/Tensordot/free:output:0@private__multi_head_self_attention/dense/Tensordot/axes:output:0Gprivate__multi_head_self_attention/dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:ô
8private__multi_head_self_attention/dense/Tensordot/stackPack@private__multi_head_self_attention/dense/Tensordot/Prod:output:0Bprivate__multi_head_self_attention/dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:Ì
<private__multi_head_self_attention/dense/Tensordot/transpose	TransposeinputsBprivate__multi_head_self_attention/dense/Tensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
:private__multi_head_self_attention/dense/Tensordot/ReshapeReshape@private__multi_head_self_attention/dense/Tensordot/transpose:y:0Aprivate__multi_head_self_attention/dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
9private__multi_head_self_attention/dense/Tensordot/MatMulMatMulCprivate__multi_head_self_attention/dense/Tensordot/Reshape:output:0Iprivate__multi_head_self_attention/dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
:private__multi_head_self_attention/dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 
@private__multi_head_self_attention/dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ë
;private__multi_head_self_attention/dense/Tensordot/concat_1ConcatV2Dprivate__multi_head_self_attention/dense/Tensordot/GatherV2:output:0Cprivate__multi_head_self_attention/dense/Tensordot/Const_2:output:0Iprivate__multi_head_self_attention/dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:ÿ
2private__multi_head_self_attention/dense/TensordotReshapeCprivate__multi_head_self_attention/dense/Tensordot/MatMul:product:0Dprivate__multi_head_self_attention/dense/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ä
?private__multi_head_self_attention/dense/BiasAdd/ReadVariableOpReadVariableOpHprivate__multi_head_self_attention_dense_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0ø
0private__multi_head_self_attention/dense/BiasAddBiasAdd;private__multi_head_self_attention/dense/Tensordot:output:0Gprivate__multi_head_self_attention/dense/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ð
Cprivate__multi_head_self_attention/dense_1/Tensordot/ReadVariableOpReadVariableOpLprivate__multi_head_self_attention_dense_1_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype0
9private__multi_head_self_attention/dense_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:
9private__multi_head_self_attention/dense_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       p
:private__multi_head_self_attention/dense_1/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:
Bprivate__multi_head_self_attention/dense_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ç
=private__multi_head_self_attention/dense_1/Tensordot/GatherV2GatherV2Cprivate__multi_head_self_attention/dense_1/Tensordot/Shape:output:0Bprivate__multi_head_self_attention/dense_1/Tensordot/free:output:0Kprivate__multi_head_self_attention/dense_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:
Dprivate__multi_head_self_attention/dense_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ë
?private__multi_head_self_attention/dense_1/Tensordot/GatherV2_1GatherV2Cprivate__multi_head_self_attention/dense_1/Tensordot/Shape:output:0Bprivate__multi_head_self_attention/dense_1/Tensordot/axes:output:0Mprivate__multi_head_self_attention/dense_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:
:private__multi_head_self_attention/dense_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ï
9private__multi_head_self_attention/dense_1/Tensordot/ProdProdFprivate__multi_head_self_attention/dense_1/Tensordot/GatherV2:output:0Cprivate__multi_head_self_attention/dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: 
<private__multi_head_self_attention/dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: õ
;private__multi_head_self_attention/dense_1/Tensordot/Prod_1ProdHprivate__multi_head_self_attention/dense_1/Tensordot/GatherV2_1:output:0Eprivate__multi_head_self_attention/dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 
@private__multi_head_self_attention/dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : È
;private__multi_head_self_attention/dense_1/Tensordot/concatConcatV2Bprivate__multi_head_self_attention/dense_1/Tensordot/free:output:0Bprivate__multi_head_self_attention/dense_1/Tensordot/axes:output:0Iprivate__multi_head_self_attention/dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:ú
:private__multi_head_self_attention/dense_1/Tensordot/stackPackBprivate__multi_head_self_attention/dense_1/Tensordot/Prod:output:0Dprivate__multi_head_self_attention/dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:Ð
>private__multi_head_self_attention/dense_1/Tensordot/transpose	TransposeinputsDprivate__multi_head_self_attention/dense_1/Tensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
<private__multi_head_self_attention/dense_1/Tensordot/ReshapeReshapeBprivate__multi_head_self_attention/dense_1/Tensordot/transpose:y:0Cprivate__multi_head_self_attention/dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
;private__multi_head_self_attention/dense_1/Tensordot/MatMulMatMulEprivate__multi_head_self_attention/dense_1/Tensordot/Reshape:output:0Kprivate__multi_head_self_attention/dense_1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
<private__multi_head_self_attention/dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 
Bprivate__multi_head_self_attention/dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ó
=private__multi_head_self_attention/dense_1/Tensordot/concat_1ConcatV2Fprivate__multi_head_self_attention/dense_1/Tensordot/GatherV2:output:0Eprivate__multi_head_self_attention/dense_1/Tensordot/Const_2:output:0Kprivate__multi_head_self_attention/dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
4private__multi_head_self_attention/dense_1/TensordotReshapeEprivate__multi_head_self_attention/dense_1/Tensordot/MatMul:product:0Fprivate__multi_head_self_attention/dense_1/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ È
Aprivate__multi_head_self_attention/dense_1/BiasAdd/ReadVariableOpReadVariableOpJprivate__multi_head_self_attention_dense_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0þ
2private__multi_head_self_attention/dense_1/BiasAddBiasAdd=private__multi_head_self_attention/dense_1/Tensordot:output:0Iprivate__multi_head_self_attention/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ð
Cprivate__multi_head_self_attention/dense_2/Tensordot/ReadVariableOpReadVariableOpLprivate__multi_head_self_attention_dense_2_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype0
9private__multi_head_self_attention/dense_2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:
9private__multi_head_self_attention/dense_2/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       p
:private__multi_head_self_attention/dense_2/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:
Bprivate__multi_head_self_attention/dense_2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ç
=private__multi_head_self_attention/dense_2/Tensordot/GatherV2GatherV2Cprivate__multi_head_self_attention/dense_2/Tensordot/Shape:output:0Bprivate__multi_head_self_attention/dense_2/Tensordot/free:output:0Kprivate__multi_head_self_attention/dense_2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:
Dprivate__multi_head_self_attention/dense_2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ë
?private__multi_head_self_attention/dense_2/Tensordot/GatherV2_1GatherV2Cprivate__multi_head_self_attention/dense_2/Tensordot/Shape:output:0Bprivate__multi_head_self_attention/dense_2/Tensordot/axes:output:0Mprivate__multi_head_self_attention/dense_2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:
:private__multi_head_self_attention/dense_2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ï
9private__multi_head_self_attention/dense_2/Tensordot/ProdProdFprivate__multi_head_self_attention/dense_2/Tensordot/GatherV2:output:0Cprivate__multi_head_self_attention/dense_2/Tensordot/Const:output:0*
T0*
_output_shapes
: 
<private__multi_head_self_attention/dense_2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: õ
;private__multi_head_self_attention/dense_2/Tensordot/Prod_1ProdHprivate__multi_head_self_attention/dense_2/Tensordot/GatherV2_1:output:0Eprivate__multi_head_self_attention/dense_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 
@private__multi_head_self_attention/dense_2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : È
;private__multi_head_self_attention/dense_2/Tensordot/concatConcatV2Bprivate__multi_head_self_attention/dense_2/Tensordot/free:output:0Bprivate__multi_head_self_attention/dense_2/Tensordot/axes:output:0Iprivate__multi_head_self_attention/dense_2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:ú
:private__multi_head_self_attention/dense_2/Tensordot/stackPackBprivate__multi_head_self_attention/dense_2/Tensordot/Prod:output:0Dprivate__multi_head_self_attention/dense_2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:Ð
>private__multi_head_self_attention/dense_2/Tensordot/transpose	TransposeinputsDprivate__multi_head_self_attention/dense_2/Tensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
<private__multi_head_self_attention/dense_2/Tensordot/ReshapeReshapeBprivate__multi_head_self_attention/dense_2/Tensordot/transpose:y:0Cprivate__multi_head_self_attention/dense_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
;private__multi_head_self_attention/dense_2/Tensordot/MatMulMatMulEprivate__multi_head_self_attention/dense_2/Tensordot/Reshape:output:0Kprivate__multi_head_self_attention/dense_2/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
<private__multi_head_self_attention/dense_2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 
Bprivate__multi_head_self_attention/dense_2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ó
=private__multi_head_self_attention/dense_2/Tensordot/concat_1ConcatV2Fprivate__multi_head_self_attention/dense_2/Tensordot/GatherV2:output:0Eprivate__multi_head_self_attention/dense_2/Tensordot/Const_2:output:0Kprivate__multi_head_self_attention/dense_2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
4private__multi_head_self_attention/dense_2/TensordotReshapeEprivate__multi_head_self_attention/dense_2/Tensordot/MatMul:product:0Fprivate__multi_head_self_attention/dense_2/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ È
Aprivate__multi_head_self_attention/dense_2/BiasAdd/ReadVariableOpReadVariableOpJprivate__multi_head_self_attention_dense_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0þ
2private__multi_head_self_attention/dense_2/BiasAddBiasAdd=private__multi_head_self_attention/dense_2/Tensordot:output:0Iprivate__multi_head_self_attention/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ }
2private__multi_head_self_attention/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿt
2private__multi_head_self_attention/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :t
2private__multi_head_self_attention/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :Ø
0private__multi_head_self_attention/Reshape/shapePack9private__multi_head_self_attention/strided_slice:output:0;private__multi_head_self_attention/Reshape/shape/1:output:0;private__multi_head_self_attention/Reshape/shape/2:output:0;private__multi_head_self_attention/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:î
*private__multi_head_self_attention/ReshapeReshape9private__multi_head_self_attention/dense/BiasAdd:output:09private__multi_head_self_attention/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
1private__multi_head_self_attention/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             í
,private__multi_head_self_attention/transpose	Transpose3private__multi_head_self_attention/Reshape:output:0:private__multi_head_self_attention/transpose/perm:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
4private__multi_head_self_attention/Reshape_1/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿv
4private__multi_head_self_attention/Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :v
4private__multi_head_self_attention/Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :à
2private__multi_head_self_attention/Reshape_1/shapePack9private__multi_head_self_attention/strided_slice:output:0=private__multi_head_self_attention/Reshape_1/shape/1:output:0=private__multi_head_self_attention/Reshape_1/shape/2:output:0=private__multi_head_self_attention/Reshape_1/shape/3:output:0*
N*
T0*
_output_shapes
:ô
,private__multi_head_self_attention/Reshape_1Reshape;private__multi_head_self_attention/dense_1/BiasAdd:output:0;private__multi_head_self_attention/Reshape_1/shape:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
3private__multi_head_self_attention/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             ó
.private__multi_head_self_attention/transpose_1	Transpose5private__multi_head_self_attention/Reshape_1:output:0<private__multi_head_self_attention/transpose_1/perm:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
4private__multi_head_self_attention/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿv
4private__multi_head_self_attention/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :v
4private__multi_head_self_attention/Reshape_2/shape/3Const*
_output_shapes
: *
dtype0*
value	B :à
2private__multi_head_self_attention/Reshape_2/shapePack9private__multi_head_self_attention/strided_slice:output:0=private__multi_head_self_attention/Reshape_2/shape/1:output:0=private__multi_head_self_attention/Reshape_2/shape/2:output:0=private__multi_head_self_attention/Reshape_2/shape/3:output:0*
N*
T0*
_output_shapes
:ô
,private__multi_head_self_attention/Reshape_2Reshape;private__multi_head_self_attention/dense_2/BiasAdd:output:0;private__multi_head_self_attention/Reshape_2/shape:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
3private__multi_head_self_attention/transpose_2/permConst*
_output_shapes
:*
dtype0*%
valueB"             ó
.private__multi_head_self_attention/transpose_2	Transpose5private__multi_head_self_attention/Reshape_2:output:0<private__multi_head_self_attention/transpose_2/perm:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿù
)private__multi_head_self_attention/MatMulBatchMatMulV20private__multi_head_self_attention/transpose:y:02private__multi_head_self_attention/transpose_1:y:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
adj_y(
*private__multi_head_self_attention/Shape_1Shape2private__multi_head_self_attention/transpose_1:y:0*
T0*
_output_shapes
:
8private__multi_head_self_attention/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ
:private__multi_head_self_attention/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
:private__multi_head_self_attention/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
2private__multi_head_self_attention/strided_slice_1StridedSlice3private__multi_head_self_attention/Shape_1:output:0Aprivate__multi_head_self_attention/strided_slice_1/stack:output:0Cprivate__multi_head_self_attention/strided_slice_1/stack_1:output:0Cprivate__multi_head_self_attention/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
'private__multi_head_self_attention/CastCast;private__multi_head_self_attention/strided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: }
'private__multi_head_self_attention/SqrtSqrt+private__multi_head_self_attention/Cast:y:0*
T0*
_output_shapes
: â
*private__multi_head_self_attention/truedivRealDiv2private__multi_head_self_attention/MatMul:output:0+private__multi_head_self_attention/Sqrt:y:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ±
*private__multi_head_self_attention/SoftmaxSoftmax.private__multi_head_self_attention/truediv:z:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿé
+private__multi_head_self_attention/MatMul_1BatchMatMulV24private__multi_head_self_attention/Softmax:softmax:02private__multi_head_self_attention/transpose_2:y:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
3private__multi_head_self_attention/transpose_3/permConst*
_output_shapes
:*
dtype0*%
valueB"             ò
.private__multi_head_self_attention/transpose_3	Transpose4private__multi_head_self_attention/MatMul_1:output:0<private__multi_head_self_attention/transpose_3/perm:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
4private__multi_head_self_attention/Reshape_3/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿv
4private__multi_head_self_attention/Reshape_3/shape/2Const*
_output_shapes
: *
dtype0*
value	B : ¡
2private__multi_head_self_attention/Reshape_3/shapePack9private__multi_head_self_attention/strided_slice:output:0=private__multi_head_self_attention/Reshape_3/shape/1:output:0=private__multi_head_self_attention/Reshape_3/shape/2:output:0*
N*
T0*
_output_shapes
:ç
,private__multi_head_self_attention/Reshape_3Reshape2private__multi_head_self_attention/transpose_3:y:0;private__multi_head_self_attention/Reshape_3/shape:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ Ð
Cprivate__multi_head_self_attention/dense_3/Tensordot/ReadVariableOpReadVariableOpLprivate__multi_head_self_attention_dense_3_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype0
9private__multi_head_self_attention/dense_3/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:
9private__multi_head_self_attention/dense_3/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       
:private__multi_head_self_attention/dense_3/Tensordot/ShapeShape5private__multi_head_self_attention/Reshape_3:output:0*
T0*
_output_shapes
:
Bprivate__multi_head_self_attention/dense_3/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ç
=private__multi_head_self_attention/dense_3/Tensordot/GatherV2GatherV2Cprivate__multi_head_self_attention/dense_3/Tensordot/Shape:output:0Bprivate__multi_head_self_attention/dense_3/Tensordot/free:output:0Kprivate__multi_head_self_attention/dense_3/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:
Dprivate__multi_head_self_attention/dense_3/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ë
?private__multi_head_self_attention/dense_3/Tensordot/GatherV2_1GatherV2Cprivate__multi_head_self_attention/dense_3/Tensordot/Shape:output:0Bprivate__multi_head_self_attention/dense_3/Tensordot/axes:output:0Mprivate__multi_head_self_attention/dense_3/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:
:private__multi_head_self_attention/dense_3/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ï
9private__multi_head_self_attention/dense_3/Tensordot/ProdProdFprivate__multi_head_self_attention/dense_3/Tensordot/GatherV2:output:0Cprivate__multi_head_self_attention/dense_3/Tensordot/Const:output:0*
T0*
_output_shapes
: 
<private__multi_head_self_attention/dense_3/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: õ
;private__multi_head_self_attention/dense_3/Tensordot/Prod_1ProdHprivate__multi_head_self_attention/dense_3/Tensordot/GatherV2_1:output:0Eprivate__multi_head_self_attention/dense_3/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 
@private__multi_head_self_attention/dense_3/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : È
;private__multi_head_self_attention/dense_3/Tensordot/concatConcatV2Bprivate__multi_head_self_attention/dense_3/Tensordot/free:output:0Bprivate__multi_head_self_attention/dense_3/Tensordot/axes:output:0Iprivate__multi_head_self_attention/dense_3/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:ú
:private__multi_head_self_attention/dense_3/Tensordot/stackPackBprivate__multi_head_self_attention/dense_3/Tensordot/Prod:output:0Dprivate__multi_head_self_attention/dense_3/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:
>private__multi_head_self_attention/dense_3/Tensordot/transpose	Transpose5private__multi_head_self_attention/Reshape_3:output:0Dprivate__multi_head_self_attention/dense_3/Tensordot/concat:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
<private__multi_head_self_attention/dense_3/Tensordot/ReshapeReshapeBprivate__multi_head_self_attention/dense_3/Tensordot/transpose:y:0Cprivate__multi_head_self_attention/dense_3/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
;private__multi_head_self_attention/dense_3/Tensordot/MatMulMatMulEprivate__multi_head_self_attention/dense_3/Tensordot/Reshape:output:0Kprivate__multi_head_self_attention/dense_3/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
<private__multi_head_self_attention/dense_3/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 
Bprivate__multi_head_self_attention/dense_3/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ó
=private__multi_head_self_attention/dense_3/Tensordot/concat_1ConcatV2Fprivate__multi_head_self_attention/dense_3/Tensordot/GatherV2:output:0Eprivate__multi_head_self_attention/dense_3/Tensordot/Const_2:output:0Kprivate__multi_head_self_attention/dense_3/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
4private__multi_head_self_attention/dense_3/TensordotReshapeEprivate__multi_head_self_attention/dense_3/Tensordot/MatMul:product:0Fprivate__multi_head_self_attention/dense_3/Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ È
Aprivate__multi_head_self_attention/dense_3/BiasAdd/ReadVariableOpReadVariableOpJprivate__multi_head_self_attention_dense_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
2private__multi_head_self_attention/dense_3/BiasAddBiasAdd=private__multi_head_self_attention/dense_3/Tensordot:output:0Iprivate__multi_head_self_attention/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
dropout/IdentityIdentity;private__multi_head_self_attention/dense_3/BiasAdd:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ f
addAddV2inputsdropout/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ |
2layer_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:¶
 layer_normalization/moments/meanMeanadd:z:0;layer_normalization/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(
(layer_normalization/moments/StopGradientStopGradient)layer_normalization/moments/mean:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿµ
-layer_normalization/moments/SquaredDifferenceSquaredDifferenceadd:z:01layer_normalization/moments/StopGradient:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
6layer_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:è
$layer_normalization/moments/varianceMean1layer_normalization/moments/SquaredDifference:z:0?layer_normalization/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(h
#layer_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½75¾
!layer_normalization/batchnorm/addAddV2-layer_normalization/moments/variance:output:0,layer_normalization/batchnorm/add/y:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#layer_normalization/batchnorm/RsqrtRsqrt%layer_normalization/batchnorm/add:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¦
0layer_normalization/batchnorm/mul/ReadVariableOpReadVariableOp9layer_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype0Â
!layer_normalization/batchnorm/mulMul'layer_normalization/batchnorm/Rsqrt:y:08layer_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
#layer_normalization/batchnorm/mul_1Muladd:z:0%layer_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ³
#layer_normalization/batchnorm/mul_2Mul)layer_normalization/moments/mean:output:0%layer_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
,layer_normalization/batchnorm/ReadVariableOpReadVariableOp5layer_normalization_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype0¾
!layer_normalization/batchnorm/subSub4layer_normalization/batchnorm/ReadVariableOp:value:0'layer_normalization/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ³
#layer_normalization/batchnorm/add_1AddV2'layer_normalization/batchnorm/mul_1:z:0%layer_normalization/batchnorm/sub:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
+sequential/dense_4/Tensordot/ReadVariableOpReadVariableOp4sequential_dense_4_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype0k
!sequential/dense_4/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:r
!sequential/dense_4/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       y
"sequential/dense_4/Tensordot/ShapeShape'layer_normalization/batchnorm/add_1:z:0*
T0*
_output_shapes
:l
*sequential/dense_4/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
%sequential/dense_4/Tensordot/GatherV2GatherV2+sequential/dense_4/Tensordot/Shape:output:0*sequential/dense_4/Tensordot/free:output:03sequential/dense_4/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:n
,sequential/dense_4/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
'sequential/dense_4/Tensordot/GatherV2_1GatherV2+sequential/dense_4/Tensordot/Shape:output:0*sequential/dense_4/Tensordot/axes:output:05sequential/dense_4/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:l
"sequential/dense_4/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: §
!sequential/dense_4/Tensordot/ProdProd.sequential/dense_4/Tensordot/GatherV2:output:0+sequential/dense_4/Tensordot/Const:output:0*
T0*
_output_shapes
: n
$sequential/dense_4/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ­
#sequential/dense_4/Tensordot/Prod_1Prod0sequential/dense_4/Tensordot/GatherV2_1:output:0-sequential/dense_4/Tensordot/Const_1:output:0*
T0*
_output_shapes
: j
(sequential/dense_4/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : è
#sequential/dense_4/Tensordot/concatConcatV2*sequential/dense_4/Tensordot/free:output:0*sequential/dense_4/Tensordot/axes:output:01sequential/dense_4/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:²
"sequential/dense_4/Tensordot/stackPack*sequential/dense_4/Tensordot/Prod:output:0,sequential/dense_4/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:Á
&sequential/dense_4/Tensordot/transpose	Transpose'layer_normalization/batchnorm/add_1:z:0,sequential/dense_4/Tensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ã
$sequential/dense_4/Tensordot/ReshapeReshape*sequential/dense_4/Tensordot/transpose:y:0+sequential/dense_4/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÃ
#sequential/dense_4/Tensordot/MatMulMatMul-sequential/dense_4/Tensordot/Reshape:output:03sequential/dense_4/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ n
$sequential/dense_4/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: l
*sequential/dense_4/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ó
%sequential/dense_4/Tensordot/concat_1ConcatV2.sequential/dense_4/Tensordot/GatherV2:output:0-sequential/dense_4/Tensordot/Const_2:output:03sequential/dense_4/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:½
sequential/dense_4/TensordotReshape-sequential/dense_4/Tensordot/MatMul:product:0.sequential/dense_4/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
)sequential/dense_4/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0¶
sequential/dense_4/BiasAddBiasAdd%sequential/dense_4/Tensordot:output:01sequential/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ {
sequential/dense_4/ReluRelu#sequential/dense_4/BiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
+sequential/dense_5/Tensordot/ReadVariableOpReadVariableOp4sequential_dense_5_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype0k
!sequential/dense_5/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:r
!sequential/dense_5/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       w
"sequential/dense_5/Tensordot/ShapeShape%sequential/dense_4/Relu:activations:0*
T0*
_output_shapes
:l
*sequential/dense_5/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
%sequential/dense_5/Tensordot/GatherV2GatherV2+sequential/dense_5/Tensordot/Shape:output:0*sequential/dense_5/Tensordot/free:output:03sequential/dense_5/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:n
,sequential/dense_5/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
'sequential/dense_5/Tensordot/GatherV2_1GatherV2+sequential/dense_5/Tensordot/Shape:output:0*sequential/dense_5/Tensordot/axes:output:05sequential/dense_5/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:l
"sequential/dense_5/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: §
!sequential/dense_5/Tensordot/ProdProd.sequential/dense_5/Tensordot/GatherV2:output:0+sequential/dense_5/Tensordot/Const:output:0*
T0*
_output_shapes
: n
$sequential/dense_5/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ­
#sequential/dense_5/Tensordot/Prod_1Prod0sequential/dense_5/Tensordot/GatherV2_1:output:0-sequential/dense_5/Tensordot/Const_1:output:0*
T0*
_output_shapes
: j
(sequential/dense_5/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : è
#sequential/dense_5/Tensordot/concatConcatV2*sequential/dense_5/Tensordot/free:output:0*sequential/dense_5/Tensordot/axes:output:01sequential/dense_5/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:²
"sequential/dense_5/Tensordot/stackPack*sequential/dense_5/Tensordot/Prod:output:0,sequential/dense_5/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:¿
&sequential/dense_5/Tensordot/transpose	Transpose%sequential/dense_4/Relu:activations:0,sequential/dense_5/Tensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ã
$sequential/dense_5/Tensordot/ReshapeReshape*sequential/dense_5/Tensordot/transpose:y:0+sequential/dense_5/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÃ
#sequential/dense_5/Tensordot/MatMulMatMul-sequential/dense_5/Tensordot/Reshape:output:03sequential/dense_5/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ n
$sequential/dense_5/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: l
*sequential/dense_5/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ó
%sequential/dense_5/Tensordot/concat_1ConcatV2.sequential/dense_5/Tensordot/GatherV2:output:0-sequential/dense_5/Tensordot/Const_2:output:03sequential/dense_5/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:½
sequential/dense_5/TensordotReshape-sequential/dense_5/Tensordot/MatMul:product:0.sequential/dense_5/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
)sequential/dense_5/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0¶
sequential/dense_5/BiasAddBiasAdd%sequential/dense_5/Tensordot:output:01sequential/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ z
dropout_1/IdentityIdentity#sequential/dense_5/BiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
add_1AddV2'layer_normalization/batchnorm/add_1:z:0dropout_1/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ~
4layer_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:¼
"layer_normalization_1/moments/meanMean	add_1:z:0=layer_normalization_1/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(
*layer_normalization_1/moments/StopGradientStopGradient+layer_normalization_1/moments/mean:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ»
/layer_normalization_1/moments/SquaredDifferenceSquaredDifference	add_1:z:03layer_normalization_1/moments/StopGradient:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
8layer_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:î
&layer_normalization_1/moments/varianceMean3layer_normalization_1/moments/SquaredDifference:z:0Alayer_normalization_1/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(j
%layer_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½75Ä
#layer_normalization_1/batchnorm/addAddV2/layer_normalization_1/moments/variance:output:0.layer_normalization_1/batchnorm/add/y:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%layer_normalization_1/batchnorm/RsqrtRsqrt'layer_normalization_1/batchnorm/add:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
2layer_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype0È
#layer_normalization_1/batchnorm/mulMul)layer_normalization_1/batchnorm/Rsqrt:y:0:layer_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
%layer_normalization_1/batchnorm/mul_1Mul	add_1:z:0'layer_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¹
%layer_normalization_1/batchnorm/mul_2Mul+layer_normalization_1/moments/mean:output:0'layer_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¢
.layer_normalization_1/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype0Ä
#layer_normalization_1/batchnorm/subSub6layer_normalization_1/batchnorm/ReadVariableOp:value:0)layer_normalization_1/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¹
%layer_normalization_1/batchnorm/add_1AddV2)layer_normalization_1/batchnorm/mul_1:z:0'layer_normalization_1/batchnorm/sub:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ }
IdentityIdentity)layer_normalization_1/batchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ æ
NoOpNoOp-^layer_normalization/batchnorm/ReadVariableOp1^layer_normalization/batchnorm/mul/ReadVariableOp/^layer_normalization_1/batchnorm/ReadVariableOp3^layer_normalization_1/batchnorm/mul/ReadVariableOp@^private__multi_head_self_attention/dense/BiasAdd/ReadVariableOpB^private__multi_head_self_attention/dense/Tensordot/ReadVariableOpB^private__multi_head_self_attention/dense_1/BiasAdd/ReadVariableOpD^private__multi_head_self_attention/dense_1/Tensordot/ReadVariableOpB^private__multi_head_self_attention/dense_2/BiasAdd/ReadVariableOpD^private__multi_head_self_attention/dense_2/Tensordot/ReadVariableOpB^private__multi_head_self_attention/dense_3/BiasAdd/ReadVariableOpD^private__multi_head_self_attention/dense_3/Tensordot/ReadVariableOp*^sequential/dense_4/BiasAdd/ReadVariableOp,^sequential/dense_4/Tensordot/ReadVariableOp*^sequential/dense_5/BiasAdd/ReadVariableOp,^sequential/dense_5/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:ÿÿÿÿÿÿÿÿÿ : : : : : : : : : : : : : : : : 2\
,layer_normalization/batchnorm/ReadVariableOp,layer_normalization/batchnorm/ReadVariableOp2d
0layer_normalization/batchnorm/mul/ReadVariableOp0layer_normalization/batchnorm/mul/ReadVariableOp2`
.layer_normalization_1/batchnorm/ReadVariableOp.layer_normalization_1/batchnorm/ReadVariableOp2h
2layer_normalization_1/batchnorm/mul/ReadVariableOp2layer_normalization_1/batchnorm/mul/ReadVariableOp2
?private__multi_head_self_attention/dense/BiasAdd/ReadVariableOp?private__multi_head_self_attention/dense/BiasAdd/ReadVariableOp2
Aprivate__multi_head_self_attention/dense/Tensordot/ReadVariableOpAprivate__multi_head_self_attention/dense/Tensordot/ReadVariableOp2
Aprivate__multi_head_self_attention/dense_1/BiasAdd/ReadVariableOpAprivate__multi_head_self_attention/dense_1/BiasAdd/ReadVariableOp2
Cprivate__multi_head_self_attention/dense_1/Tensordot/ReadVariableOpCprivate__multi_head_self_attention/dense_1/Tensordot/ReadVariableOp2
Aprivate__multi_head_self_attention/dense_2/BiasAdd/ReadVariableOpAprivate__multi_head_self_attention/dense_2/BiasAdd/ReadVariableOp2
Cprivate__multi_head_self_attention/dense_2/Tensordot/ReadVariableOpCprivate__multi_head_self_attention/dense_2/Tensordot/ReadVariableOp2
Aprivate__multi_head_self_attention/dense_3/BiasAdd/ReadVariableOpAprivate__multi_head_self_attention/dense_3/BiasAdd/ReadVariableOp2
Cprivate__multi_head_self_attention/dense_3/Tensordot/ReadVariableOpCprivate__multi_head_self_attention/dense_3/Tensordot/ReadVariableOp2V
)sequential/dense_4/BiasAdd/ReadVariableOp)sequential/dense_4/BiasAdd/ReadVariableOp2Z
+sequential/dense_4/Tensordot/ReadVariableOp+sequential/dense_4/Tensordot/ReadVariableOp2V
)sequential/dense_5/BiasAdd/ReadVariableOp)sequential/dense_5/BiasAdd/ReadVariableOp2Z
+sequential/dense_5/Tensordot/ReadVariableOp+sequential/dense_5/Tensordot/ReadVariableOp:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs"µ	L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*«
serving_default
<
input_11
serving_default_input_1:0ÿÿÿÿÿÿÿÿÿ;
dense_70
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:¦£
³
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer-4
layer_with_weights-2
layer-5
layer-6
layer_with_weights-3
layer-7
		variables

trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_network
"
_tf_keras_input_layer
Á
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
	token_emb
pos_emb"
_tf_keras_layer
ó
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
 att
!ffn
"
layernorm1
#
layernorm2
$dropout1
%dropout2"
_tf_keras_layer
¥
&	variables
'trainable_variables
(regularization_losses
)	keras_api
*__call__
*+&call_and_return_all_conditional_losses"
_tf_keras_layer
¼
,	variables
-trainable_variables
.regularization_losses
/	keras_api
0__call__
*1&call_and_return_all_conditional_losses
2_random_generator"
_tf_keras_layer
»
3	variables
4trainable_variables
5regularization_losses
6	keras_api
7__call__
*8&call_and_return_all_conditional_losses

9kernel
:bias"
_tf_keras_layer
¼
;	variables
<trainable_variables
=regularization_losses
>	keras_api
?__call__
*@&call_and_return_all_conditional_losses
A_random_generator"
_tf_keras_layer
»
B	variables
Ctrainable_variables
Dregularization_losses
E	keras_api
F__call__
*G&call_and_return_all_conditional_losses

Hkernel
Ibias"
_tf_keras_layer
Æ
J0
K1
L2
M3
N4
O5
P6
Q7
R8
S9
T10
U11
V12
W13
X14
Y15
Z16
[17
918
:19
H20
I21"
trackable_list_wrapper
Æ
J0
K1
L2
M3
N4
O5
P6
Q7
R8
S9
T10
U11
V12
W13
X14
Y15
Z16
[17
918
:19
H20
I21"
trackable_list_wrapper
 "
trackable_list_wrapper
Ê
\non_trainable_variables

]layers
^metrics
_layer_regularization_losses
`layer_metrics
		variables

trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
Å
atrace_0
btrace_1
ctrace_2
dtrace_32Ú
$__inference_model_layer_call_fn_5300
$__inference_model_layer_call_fn_6127
$__inference_model_layer_call_fn_6176
$__inference_model_layer_call_fn_5911¿
¶²²
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
annotationsª *
 zatrace_0zbtrace_1zctrace_2zdtrace_3
±
etrace_0
ftrace_1
gtrace_2
htrace_32Æ
?__inference_model_layer_call_and_return_conditional_losses_6458
?__inference_model_layer_call_and_return_conditional_losses_6768
?__inference_model_layer_call_and_return_conditional_losses_5966
?__inference_model_layer_call_and_return_conditional_losses_6021¿
¶²²
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
annotationsª *
 zetrace_0zftrace_1zgtrace_2zhtrace_3
ÊBÇ
__inference__wrapped_model_4682input_1"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 

iiter

jbeta_1

kbeta_2
	ldecay
mlearning_rate9mâ:mãHmäImåJmæKmçLmèMméNmêOmëPmìQmíRmîSmïTmðUmñVmòWmóXmôYmõZmö[m÷9vø:vùHvúIvûJvüKvýLvþMvÿNvOvPvQvRvSvTvUvVvWvXvYvZv[v"
	optimizer
,
nserving_default"
signature_map
.
J0
K1"
trackable_list_wrapper
.
J0
K1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
onon_trainable_variables

players
qmetrics
rlayer_regularization_losses
slayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object

ttrace_02æ
D__inference_private__token_and_position_embedding_layer_call_fn_6777
²
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
annotationsª *
 zttrace_0

utrace_02
___inference_private__token_and_position_embedding_layer_call_and_return_conditional_losses_6801
²
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
annotationsª *
 zutrace_0
µ
v	variables
wtrainable_variables
xregularization_losses
y	keras_api
z__call__
*{&call_and_return_all_conditional_losses
J
embeddings"
_tf_keras_layer
·
|	variables
}trainable_variables
~regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
K
embeddings"
_tf_keras_layer

L0
M1
N2
O3
P4
Q5
R6
S7
T8
U9
V10
W11
X12
Y13
Z14
[15"
trackable_list_wrapper

L0
M1
N2
O3
P4
Q5
R6
S7
T8
U9
V10
W11
X12
Y13
Z14
[15"
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
â
trace_0
trace_12§
9__inference_private__transformer_block_layer_call_fn_6838
9__inference_private__transformer_block_layer_call_fn_6875®
¥²¡
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0ztrace_1

trace_0
trace_12Ý
T__inference_private__transformer_block_layer_call_and_return_conditional_losses_7119
T__inference_private__transformer_block_layer_call_and_return_conditional_losses_7377®
¥²¡
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0ztrace_1
ó
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
query_dense
	key_dense
value_dense
combine_heads"
_tf_keras_layer

layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_sequential
Ë
	variables
trainable_variables
regularization_losses
 	keras_api
¡__call__
+¢&call_and_return_all_conditional_losses
	£axis
	Xgamma
Ybeta"
_tf_keras_layer
Ë
¤	variables
¥trainable_variables
¦regularization_losses
§	keras_api
¨__call__
+©&call_and_return_all_conditional_losses
	ªaxis
	Zgamma
[beta"
_tf_keras_layer
Ã
«	variables
¬trainable_variables
­regularization_losses
®	keras_api
¯__call__
+°&call_and_return_all_conditional_losses
±_random_generator"
_tf_keras_layer
Ã
²	variables
³trainable_variables
´regularization_losses
µ	keras_api
¶__call__
+·&call_and_return_all_conditional_losses
¸_random_generator"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
¹non_trainable_variables
ºlayers
»metrics
 ¼layer_regularization_losses
½layer_metrics
&	variables
'trainable_variables
(regularization_losses
*__call__
*+&call_and_return_all_conditional_losses
&+"call_and_return_conditional_losses"
_generic_user_object

¾trace_02ë
7__inference_global_average_pooling1d_layer_call_fn_7382¯
¦²¢
FullArgSpec%
args
jself
jinputs
jmask
varargs
 
varkw
 
defaults¢

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z¾trace_0
¥
¿trace_02
R__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_7388¯
¦²¢
FullArgSpec%
args
jself
jinputs
jmask
varargs
 
varkw
 
defaults¢

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z¿trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
Ànon_trainable_variables
Álayers
Âmetrics
 Ãlayer_regularization_losses
Älayer_metrics
,	variables
-trainable_variables
.regularization_losses
0__call__
*1&call_and_return_all_conditional_losses
&1"call_and_return_conditional_losses"
_generic_user_object
Å
Åtrace_0
Ætrace_12
(__inference_dropout_2_layer_call_fn_7393
(__inference_dropout_2_layer_call_fn_7398³
ª²¦
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
annotationsª *
 zÅtrace_0zÆtrace_1
û
Çtrace_0
Ètrace_12À
C__inference_dropout_2_layer_call_and_return_conditional_losses_7403
C__inference_dropout_2_layer_call_and_return_conditional_losses_7415³
ª²¦
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
annotationsª *
 zÇtrace_0zÈtrace_1
"
_generic_user_object
.
90
:1"
trackable_list_wrapper
.
90
:1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
Énon_trainable_variables
Êlayers
Ëmetrics
 Ìlayer_regularization_losses
Ílayer_metrics
3	variables
4trainable_variables
5regularization_losses
7__call__
*8&call_and_return_all_conditional_losses
&8"call_and_return_conditional_losses"
_generic_user_object
ì
Îtrace_02Í
&__inference_dense_6_layer_call_fn_7424¢
²
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
annotationsª *
 zÎtrace_0

Ïtrace_02è
A__inference_dense_6_layer_call_and_return_conditional_losses_7435¢
²
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
annotationsª *
 zÏtrace_0
 : 2dense_6/kernel
:2dense_6/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
Ðnon_trainable_variables
Ñlayers
Òmetrics
 Ólayer_regularization_losses
Ôlayer_metrics
;	variables
<trainable_variables
=regularization_losses
?__call__
*@&call_and_return_all_conditional_losses
&@"call_and_return_conditional_losses"
_generic_user_object
Å
Õtrace_0
Ötrace_12
(__inference_dropout_3_layer_call_fn_7440
(__inference_dropout_3_layer_call_fn_7445³
ª²¦
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
annotationsª *
 zÕtrace_0zÖtrace_1
û
×trace_0
Øtrace_12À
C__inference_dropout_3_layer_call_and_return_conditional_losses_7450
C__inference_dropout_3_layer_call_and_return_conditional_losses_7462³
ª²¦
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
annotationsª *
 z×trace_0zØtrace_1
"
_generic_user_object
.
H0
I1"
trackable_list_wrapper
.
H0
I1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
Ùnon_trainable_variables
Úlayers
Ûmetrics
 Ülayer_regularization_losses
Ýlayer_metrics
B	variables
Ctrainable_variables
Dregularization_losses
F__call__
*G&call_and_return_all_conditional_losses
&G"call_and_return_conditional_losses"
_generic_user_object
ì
Þtrace_02Í
&__inference_dense_7_layer_call_fn_7471¢
²
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
annotationsª *
 zÞtrace_0

ßtrace_02è
A__inference_dense_7_layer_call_and_return_conditional_losses_7482¢
²
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
annotationsª *
 zßtrace_0
 :2dense_7/kernel
:2dense_7/bias
M:K	Æ 2:private__token_and_position_embedding/embedding/embeddings
O:M	 2<private__token_and_position_embedding/embedding_1/embeddings
\:Z  2Jprivate__transformer_block/private__multi_head_self_attention/dense/kernel
V:T 2Hprivate__transformer_block/private__multi_head_self_attention/dense/bias
^:\  2Lprivate__transformer_block/private__multi_head_self_attention/dense_1/kernel
X:V 2Jprivate__transformer_block/private__multi_head_self_attention/dense_1/bias
^:\  2Lprivate__transformer_block/private__multi_head_self_attention/dense_2/kernel
X:V 2Jprivate__transformer_block/private__multi_head_self_attention/dense_2/bias
^:\  2Lprivate__transformer_block/private__multi_head_self_attention/dense_3/kernel
X:V 2Jprivate__transformer_block/private__multi_head_self_attention/dense_3/bias
 :  2dense_4/kernel
: 2dense_4/bias
 :  2dense_5/kernel
: 2dense_5/bias
B:@ 24private__transformer_block/layer_normalization/gamma
A:? 23private__transformer_block/layer_normalization/beta
D:B 26private__transformer_block/layer_normalization_1/gamma
C:A 25private__transformer_block/layer_normalization_1/beta
 "
trackable_list_wrapper
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
0
à0
á1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
öBó
$__inference_model_layer_call_fn_5300input_1"¿
¶²²
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
annotationsª *
 
õBò
$__inference_model_layer_call_fn_6127inputs"¿
¶²²
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
annotationsª *
 
õBò
$__inference_model_layer_call_fn_6176inputs"¿
¶²²
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
annotationsª *
 
öBó
$__inference_model_layer_call_fn_5911input_1"¿
¶²²
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
annotationsª *
 
B
?__inference_model_layer_call_and_return_conditional_losses_6458inputs"¿
¶²²
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
annotationsª *
 
B
?__inference_model_layer_call_and_return_conditional_losses_6768inputs"¿
¶²²
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
annotationsª *
 
B
?__inference_model_layer_call_and_return_conditional_losses_5966input_1"¿
¶²²
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
annotationsª *
 
B
?__inference_model_layer_call_and_return_conditional_losses_6021input_1"¿
¶²²
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
annotationsª *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
ÉBÆ
"__inference_signature_wrapper_6078input_1"
²
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
annotationsª *
 
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
îBë
D__inference_private__token_and_position_embedding_layer_call_fn_6777x"
²
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
annotationsª *
 
B
___inference_private__token_and_position_embedding_layer_call_and_return_conditional_losses_6801x"
²
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
annotationsª *
 
'
J0"
trackable_list_wrapper
'
J0"
trackable_list_wrapper
 "
trackable_list_wrapper
²
ânon_trainable_variables
ãlayers
ämetrics
 ålayer_regularization_losses
ælayer_metrics
v	variables
wtrainable_variables
xregularization_losses
z__call__
*{&call_and_return_all_conditional_losses
&{"call_and_return_conditional_losses"
_generic_user_object
¨2¥¢
²
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
annotationsª *
 
¨2¥¢
²
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
annotationsª *
 
'
K0"
trackable_list_wrapper
'
K0"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
çnon_trainable_variables
èlayers
émetrics
 êlayer_regularization_losses
ëlayer_metrics
|	variables
}trainable_variables
~regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
¨2¥¢
²
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
annotationsª *
 
¨2¥¢
²
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
annotationsª *
 
 "
trackable_list_wrapper
J
 0
!1
"2
#3
$4
%5"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ùBö
9__inference_private__transformer_block_layer_call_fn_6838inputs"®
¥²¡
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ùBö
9__inference_private__transformer_block_layer_call_fn_6875inputs"®
¥²¡
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
T__inference_private__transformer_block_layer_call_and_return_conditional_losses_7119inputs"®
¥²¡
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
T__inference_private__transformer_block_layer_call_and_return_conditional_losses_7377inputs"®
¥²¡
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
X
L0
M1
N2
O3
P4
Q5
R6
S7"
trackable_list_wrapper
X
L0
M1
N2
O3
P4
Q5
R6
S7"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ìnon_trainable_variables
ílayers
îmetrics
 ïlayer_regularization_losses
ðlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
¨2¥¢
²
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
annotationsª *
 
¨2¥¢
²
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
annotationsª *
 
Á
ñ	variables
òtrainable_variables
óregularization_losses
ô	keras_api
õ__call__
+ö&call_and_return_all_conditional_losses

Lkernel
Mbias"
_tf_keras_layer
Á
÷	variables
øtrainable_variables
ùregularization_losses
ú	keras_api
û__call__
+ü&call_and_return_all_conditional_losses

Nkernel
Obias"
_tf_keras_layer
Á
ý	variables
þtrainable_variables
ÿregularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses

Pkernel
Qbias"
_tf_keras_layer
Á
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses

Rkernel
Sbias"
_tf_keras_layer
Á
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses

Tkernel
Ubias"
_tf_keras_layer
Á
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses

Vkernel
Wbias"
_tf_keras_layer
<
T0
U1
V2
W3"
trackable_list_wrapper
<
T0
U1
V2
W3"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
á
trace_0
trace_1
trace_2
trace_32î
)__inference_sequential_layer_call_fn_4774
)__inference_sequential_layer_call_fn_7495
)__inference_sequential_layer_call_fn_7508
)__inference_sequential_layer_call_fn_4847¿
¶²²
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
annotationsª *
 ztrace_0ztrace_1ztrace_2ztrace_3
Í
trace_0
trace_1
 trace_2
¡trace_32Ú
D__inference_sequential_layer_call_and_return_conditional_losses_7565
D__inference_sequential_layer_call_and_return_conditional_losses_7622
D__inference_sequential_layer_call_and_return_conditional_losses_4861
D__inference_sequential_layer_call_and_return_conditional_losses_4875¿
¶²²
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
annotationsª *
 ztrace_0ztrace_1z trace_2z¡trace_3
.
X0
Y1"
trackable_list_wrapper
.
X0
Y1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
¢non_trainable_variables
£layers
¤metrics
 ¥layer_regularization_losses
¦layer_metrics
	variables
trainable_variables
regularization_losses
¡__call__
+¢&call_and_return_all_conditional_losses
'¢"call_and_return_conditional_losses"
_generic_user_object
¨2¥¢
²
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
annotationsª *
 
¨2¥¢
²
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
annotationsª *
 
 "
trackable_list_wrapper
.
Z0
[1"
trackable_list_wrapper
.
Z0
[1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
§non_trainable_variables
¨layers
©metrics
 ªlayer_regularization_losses
«layer_metrics
¤	variables
¥trainable_variables
¦regularization_losses
¨__call__
+©&call_and_return_all_conditional_losses
'©"call_and_return_conditional_losses"
_generic_user_object
¨2¥¢
²
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
annotationsª *
 
¨2¥¢
²
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
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
¬non_trainable_variables
­layers
®metrics
 ¯layer_regularization_losses
°layer_metrics
«	variables
¬trainable_variables
­regularization_losses
¯__call__
+°&call_and_return_all_conditional_losses
'°"call_and_return_conditional_losses"
_generic_user_object
¹2¶³
ª²¦
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
annotationsª *
 
¹2¶³
ª²¦
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
annotationsª *
 
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
±non_trainable_variables
²layers
³metrics
 ´layer_regularization_losses
µlayer_metrics
²	variables
³trainable_variables
´regularization_losses
¶__call__
+·&call_and_return_all_conditional_losses
'·"call_and_return_conditional_losses"
_generic_user_object
¹2¶³
ª²¦
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
annotationsª *
 
¹2¶³
ª²¦
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
annotationsª *
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
øBõ
7__inference_global_average_pooling1d_layer_call_fn_7382inputs"¯
¦²¢
FullArgSpec%
args
jself
jinputs
jmask
varargs
 
varkw
 
defaults¢

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
R__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_7388inputs"¯
¦²¢
FullArgSpec%
args
jself
jinputs
jmask
varargs
 
varkw
 
defaults¢

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
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
íBê
(__inference_dropout_2_layer_call_fn_7393inputs"³
ª²¦
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
annotationsª *
 
íBê
(__inference_dropout_2_layer_call_fn_7398inputs"³
ª²¦
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
annotationsª *
 
B
C__inference_dropout_2_layer_call_and_return_conditional_losses_7403inputs"³
ª²¦
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
annotationsª *
 
B
C__inference_dropout_2_layer_call_and_return_conditional_losses_7415inputs"³
ª²¦
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
annotationsª *
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
ÚB×
&__inference_dense_6_layer_call_fn_7424inputs"¢
²
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
annotationsª *
 
õBò
A__inference_dense_6_layer_call_and_return_conditional_losses_7435inputs"¢
²
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
annotationsª *
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
íBê
(__inference_dropout_3_layer_call_fn_7440inputs"³
ª²¦
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
annotationsª *
 
íBê
(__inference_dropout_3_layer_call_fn_7445inputs"³
ª²¦
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
annotationsª *
 
B
C__inference_dropout_3_layer_call_and_return_conditional_losses_7450inputs"³
ª²¦
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
annotationsª *
 
B
C__inference_dropout_3_layer_call_and_return_conditional_losses_7462inputs"³
ª²¦
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
annotationsª *
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
ÚB×
&__inference_dense_7_layer_call_fn_7471inputs"¢
²
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
annotationsª *
 
õBò
A__inference_dense_7_layer_call_and_return_conditional_losses_7482inputs"¢
²
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
annotationsª *
 
R
¶	variables
·	keras_api

¸total

¹count"
_tf_keras_metric

º	variables
»	keras_api
¼true_positives
½true_negatives
¾false_positives
¿false_negatives"
_tf_keras_metric
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
 "
trackable_list_wrapper
@
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
L0
M1"
trackable_list_wrapper
.
L0
M1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Ànon_trainable_variables
Álayers
Âmetrics
 Ãlayer_regularization_losses
Älayer_metrics
ñ	variables
òtrainable_variables
óregularization_losses
õ__call__
+ö&call_and_return_all_conditional_losses
'ö"call_and_return_conditional_losses"
_generic_user_object
¨2¥¢
²
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
annotationsª *
 
¨2¥¢
²
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
annotationsª *
 
.
N0
O1"
trackable_list_wrapper
.
N0
O1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Ånon_trainable_variables
Ælayers
Çmetrics
 Èlayer_regularization_losses
Élayer_metrics
÷	variables
øtrainable_variables
ùregularization_losses
û__call__
+ü&call_and_return_all_conditional_losses
'ü"call_and_return_conditional_losses"
_generic_user_object
¨2¥¢
²
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
annotationsª *
 
¨2¥¢
²
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
annotationsª *
 
.
P0
Q1"
trackable_list_wrapper
.
P0
Q1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Ênon_trainable_variables
Ëlayers
Ìmetrics
 Ílayer_regularization_losses
Îlayer_metrics
ý	variables
þtrainable_variables
ÿregularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
¨2¥¢
²
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
annotationsª *
 
¨2¥¢
²
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
annotationsª *
 
.
R0
S1"
trackable_list_wrapper
.
R0
S1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Ïnon_trainable_variables
Ðlayers
Ñmetrics
 Òlayer_regularization_losses
Ólayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
¨2¥¢
²
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
annotationsª *
 
¨2¥¢
²
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
annotationsª *
 
.
T0
U1"
trackable_list_wrapper
.
T0
U1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Ônon_trainable_variables
Õlayers
Ömetrics
 ×layer_regularization_losses
Ølayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
ì
Ùtrace_02Í
&__inference_dense_4_layer_call_fn_7631¢
²
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
annotationsª *
 zÙtrace_0

Útrace_02è
A__inference_dense_4_layer_call_and_return_conditional_losses_7662¢
²
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
annotationsª *
 zÚtrace_0
.
V0
W1"
trackable_list_wrapper
.
V0
W1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Ûnon_trainable_variables
Ülayers
Ýmetrics
 Þlayer_regularization_losses
ßlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
ì
àtrace_02Í
&__inference_dense_5_layer_call_fn_7671¢
²
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
annotationsª *
 zàtrace_0

átrace_02è
A__inference_dense_5_layer_call_and_return_conditional_losses_7701¢
²
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
annotationsª *
 zátrace_0
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
Bþ
)__inference_sequential_layer_call_fn_4774dense_4_input"¿
¶²²
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
annotationsª *
 
úB÷
)__inference_sequential_layer_call_fn_7495inputs"¿
¶²²
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
annotationsª *
 
úB÷
)__inference_sequential_layer_call_fn_7508inputs"¿
¶²²
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
annotationsª *
 
Bþ
)__inference_sequential_layer_call_fn_4847dense_4_input"¿
¶²²
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
annotationsª *
 
B
D__inference_sequential_layer_call_and_return_conditional_losses_7565inputs"¿
¶²²
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
annotationsª *
 
B
D__inference_sequential_layer_call_and_return_conditional_losses_7622inputs"¿
¶²²
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
annotationsª *
 
B
D__inference_sequential_layer_call_and_return_conditional_losses_4861dense_4_input"¿
¶²²
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
annotationsª *
 
B
D__inference_sequential_layer_call_and_return_conditional_losses_4875dense_4_input"¿
¶²²
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
annotationsª *
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
0
¸0
¹1"
trackable_list_wrapper
.
¶	variables"
_generic_user_object
:  (2total
:  (2count
@
¼0
½1
¾2
¿3"
trackable_list_wrapper
.
º	variables"
_generic_user_object
:È (2true_positives
:È (2true_negatives
 :È (2false_positives
 :È (2false_negatives
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
ÚB×
&__inference_dense_4_layer_call_fn_7631inputs"¢
²
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
annotationsª *
 
õBò
A__inference_dense_4_layer_call_and_return_conditional_losses_7662inputs"¢
²
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
annotationsª *
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
ÚB×
&__inference_dense_5_layer_call_fn_7671inputs"¢
²
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
annotationsª *
 
õBò
A__inference_dense_5_layer_call_and_return_conditional_losses_7701inputs"¢
²
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
annotationsª *
 
%:# 2Adam/dense_6/kernel/m
:2Adam/dense_6/bias/m
%:#2Adam/dense_7/kernel/m
:2Adam/dense_7/bias/m
R:P	Æ 2AAdam/private__token_and_position_embedding/embedding/embeddings/m
T:R	 2CAdam/private__token_and_position_embedding/embedding_1/embeddings/m
a:_  2QAdam/private__transformer_block/private__multi_head_self_attention/dense/kernel/m
[:Y 2OAdam/private__transformer_block/private__multi_head_self_attention/dense/bias/m
c:a  2SAdam/private__transformer_block/private__multi_head_self_attention/dense_1/kernel/m
]:[ 2QAdam/private__transformer_block/private__multi_head_self_attention/dense_1/bias/m
c:a  2SAdam/private__transformer_block/private__multi_head_self_attention/dense_2/kernel/m
]:[ 2QAdam/private__transformer_block/private__multi_head_self_attention/dense_2/bias/m
c:a  2SAdam/private__transformer_block/private__multi_head_self_attention/dense_3/kernel/m
]:[ 2QAdam/private__transformer_block/private__multi_head_self_attention/dense_3/bias/m
%:#  2Adam/dense_4/kernel/m
: 2Adam/dense_4/bias/m
%:#  2Adam/dense_5/kernel/m
: 2Adam/dense_5/bias/m
G:E 2;Adam/private__transformer_block/layer_normalization/gamma/m
F:D 2:Adam/private__transformer_block/layer_normalization/beta/m
I:G 2=Adam/private__transformer_block/layer_normalization_1/gamma/m
H:F 2<Adam/private__transformer_block/layer_normalization_1/beta/m
%:# 2Adam/dense_6/kernel/v
:2Adam/dense_6/bias/v
%:#2Adam/dense_7/kernel/v
:2Adam/dense_7/bias/v
R:P	Æ 2AAdam/private__token_and_position_embedding/embedding/embeddings/v
T:R	 2CAdam/private__token_and_position_embedding/embedding_1/embeddings/v
a:_  2QAdam/private__transformer_block/private__multi_head_self_attention/dense/kernel/v
[:Y 2OAdam/private__transformer_block/private__multi_head_self_attention/dense/bias/v
c:a  2SAdam/private__transformer_block/private__multi_head_self_attention/dense_1/kernel/v
]:[ 2QAdam/private__transformer_block/private__multi_head_self_attention/dense_1/bias/v
c:a  2SAdam/private__transformer_block/private__multi_head_self_attention/dense_2/kernel/v
]:[ 2QAdam/private__transformer_block/private__multi_head_self_attention/dense_2/bias/v
c:a  2SAdam/private__transformer_block/private__multi_head_self_attention/dense_3/kernel/v
]:[ 2QAdam/private__transformer_block/private__multi_head_self_attention/dense_3/bias/v
%:#  2Adam/dense_4/kernel/v
: 2Adam/dense_4/bias/v
%:#  2Adam/dense_5/kernel/v
: 2Adam/dense_5/bias/v
G:E 2;Adam/private__transformer_block/layer_normalization/gamma/v
F:D 2:Adam/private__transformer_block/layer_normalization/beta/v
I:G 2=Adam/private__transformer_block/layer_normalization_1/gamma/v
H:F 2<Adam/private__transformer_block/layer_normalization_1/beta/v¡
__inference__wrapped_model_4682~KJLMNOPQRSXYTUVWZ[9:HI1¢.
'¢$
"
input_1ÿÿÿÿÿÿÿÿÿ
ª "1ª.
,
dense_7!
dense_7ÿÿÿÿÿÿÿÿÿ«
A__inference_dense_4_layer_call_and_return_conditional_losses_7662fTU4¢1
*¢'
%"
inputsÿÿÿÿÿÿÿÿÿ 
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿ 
 
&__inference_dense_4_layer_call_fn_7631YTU4¢1
*¢'
%"
inputsÿÿÿÿÿÿÿÿÿ 
ª "ÿÿÿÿÿÿÿÿÿ «
A__inference_dense_5_layer_call_and_return_conditional_losses_7701fVW4¢1
*¢'
%"
inputsÿÿÿÿÿÿÿÿÿ 
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿ 
 
&__inference_dense_5_layer_call_fn_7671YVW4¢1
*¢'
%"
inputsÿÿÿÿÿÿÿÿÿ 
ª "ÿÿÿÿÿÿÿÿÿ ¡
A__inference_dense_6_layer_call_and_return_conditional_losses_7435\9:/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 y
&__inference_dense_6_layer_call_fn_7424O9:/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ 
ª "ÿÿÿÿÿÿÿÿÿ¡
A__inference_dense_7_layer_call_and_return_conditional_losses_7482\HI/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 y
&__inference_dense_7_layer_call_fn_7471OHI/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ£
C__inference_dropout_2_layer_call_and_return_conditional_losses_7403\3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ 
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ 
 £
C__inference_dropout_2_layer_call_and_return_conditional_losses_7415\3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ 
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ 
 {
(__inference_dropout_2_layer_call_fn_7393O3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ 
p 
ª "ÿÿÿÿÿÿÿÿÿ {
(__inference_dropout_2_layer_call_fn_7398O3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ 
p
ª "ÿÿÿÿÿÿÿÿÿ £
C__inference_dropout_3_layer_call_and_return_conditional_losses_7450\3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 £
C__inference_dropout_3_layer_call_and_return_conditional_losses_7462\3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 {
(__inference_dropout_3_layer_call_fn_7440O3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ{
(__inference_dropout_3_layer_call_fn_7445O3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿÑ
R__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_7388{I¢F
?¢<
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ©
7__inference_global_average_pooling1d_layer_call_fn_7382nI¢F
?¢<
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
ª "!ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ½
?__inference_model_layer_call_and_return_conditional_losses_5966zKJLMNOPQRSXYTUVWZ[9:HI9¢6
/¢,
"
input_1ÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ½
?__inference_model_layer_call_and_return_conditional_losses_6021zKJLMNOPQRSXYTUVWZ[9:HI9¢6
/¢,
"
input_1ÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¼
?__inference_model_layer_call_and_return_conditional_losses_6458yKJLMNOPQRSXYTUVWZ[9:HI8¢5
.¢+
!
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¼
?__inference_model_layer_call_and_return_conditional_losses_6768yKJLMNOPQRSXYTUVWZ[9:HI8¢5
.¢+
!
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
$__inference_model_layer_call_fn_5300mKJLMNOPQRSXYTUVWZ[9:HI9¢6
/¢,
"
input_1ÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
$__inference_model_layer_call_fn_5911mKJLMNOPQRSXYTUVWZ[9:HI9¢6
/¢,
"
input_1ÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ
$__inference_model_layer_call_fn_6127lKJLMNOPQRSXYTUVWZ[9:HI8¢5
.¢+
!
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
$__inference_model_layer_call_fn_6176lKJLMNOPQRSXYTUVWZ[9:HI8¢5
.¢+
!
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿÀ
___inference_private__token_and_position_embedding_layer_call_and_return_conditional_losses_6801]KJ+¢(
!¢

xÿÿÿÿÿÿÿÿÿ
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿ 
 
D__inference_private__token_and_position_embedding_layer_call_fn_6777PKJ+¢(
!¢

xÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ Ð
T__inference_private__transformer_block_layer_call_and_return_conditional_losses_7119xLMNOPQRSXYTUVWZ[8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿ 
p 
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿ 
 Ð
T__inference_private__transformer_block_layer_call_and_return_conditional_losses_7377xLMNOPQRSXYTUVWZ[8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿ 
p
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿ 
 ¨
9__inference_private__transformer_block_layer_call_fn_6838kLMNOPQRSXYTUVWZ[8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿ 
p 
ª "ÿÿÿÿÿÿÿÿÿ ¨
9__inference_private__transformer_block_layer_call_fn_6875kLMNOPQRSXYTUVWZ[8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿ 
p
ª "ÿÿÿÿÿÿÿÿÿ ¿
D__inference_sequential_layer_call_and_return_conditional_losses_4861wTUVWC¢@
9¢6
,)
dense_4_inputÿÿÿÿÿÿÿÿÿ 
p 

 
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿ 
 ¿
D__inference_sequential_layer_call_and_return_conditional_losses_4875wTUVWC¢@
9¢6
,)
dense_4_inputÿÿÿÿÿÿÿÿÿ 
p

 
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿ 
 ¸
D__inference_sequential_layer_call_and_return_conditional_losses_7565pTUVW<¢9
2¢/
%"
inputsÿÿÿÿÿÿÿÿÿ 
p 

 
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿ 
 ¸
D__inference_sequential_layer_call_and_return_conditional_losses_7622pTUVW<¢9
2¢/
%"
inputsÿÿÿÿÿÿÿÿÿ 
p

 
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿ 
 
)__inference_sequential_layer_call_fn_4774jTUVWC¢@
9¢6
,)
dense_4_inputÿÿÿÿÿÿÿÿÿ 
p 

 
ª "ÿÿÿÿÿÿÿÿÿ 
)__inference_sequential_layer_call_fn_4847jTUVWC¢@
9¢6
,)
dense_4_inputÿÿÿÿÿÿÿÿÿ 
p

 
ª "ÿÿÿÿÿÿÿÿÿ 
)__inference_sequential_layer_call_fn_7495cTUVW<¢9
2¢/
%"
inputsÿÿÿÿÿÿÿÿÿ 
p 

 
ª "ÿÿÿÿÿÿÿÿÿ 
)__inference_sequential_layer_call_fn_7508cTUVW<¢9
2¢/
%"
inputsÿÿÿÿÿÿÿÿÿ 
p

 
ª "ÿÿÿÿÿÿÿÿÿ °
"__inference_signature_wrapper_6078KJLMNOPQRSXYTUVWZ[9:HI<¢9
¢ 
2ª/
-
input_1"
input_1ÿÿÿÿÿÿÿÿÿ"1ª.
,
dense_7!
dense_7ÿÿÿÿÿÿÿÿÿ
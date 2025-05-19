# RISC-V Vectorized CNN for MNIST (28x28)

.data
.section .data

IMAGE_SIZE:     .word 28
FILTER_SIZE:    .word 5
NUM_FILTERS:    .word 8
POOL_SIZE:      .word 2
NUM_CLASSES:    .word 10

.include "input_image.s"   

.include "ConvoW_fc_riscv.s"
.include "Convob_fc_riscv.s"
.include "DenseW_fc_riscv.s"
.include "Denseb_fc_riscv.s"

conv_output:    .space 4608  
relu_output:    .space 4608 
pool_output:    .space 1152  
flattened:      .space 1152  
fc_layer_output: .space 10   
softmax_output: .space 10    

.text
.global _start

_start:
    la a0, input_image
    lw a1, IMAGE_SIZE
    lw a2, FILTER_SIZE
    lw a3, NUM_FILTERS
    
    jal ra, conv2d
    
    la a0, conv_output
    li a1, 24
    li a2, 24
    lw a3, NUM_FILTERS
    jal ra, relu
    
    la a0, relu_output
    la a1, pool_output
    li a2, 24
    lw a3, NUM_FILTERS
    lw a4, POOL_SIZE
    jal ra, max_pool2d
    
    la a0, pool_output
    la a1, flattened
    li a2, 12
    li a3, 12
    lw a4, NUM_FILTERS
    jal ra, flatten
    
    la a0, flattened
    la a1, fc_layer_output  
    la a2, fc_layer_weights
    la a3, fc_layer_biases
    li a4, 1152
    lw a5, NUM_CLASSES
    jal ra, dense_layer
    
    la a0, fc_layer_output  
    la a1, softmax_output
    lw a2, NUM_CLASSES
    jal ra, softmax
    
    li a7, 93
    ecall

conv2d:
    sub t0, a1, a2     
    addi t0, t0, 1
    
    la t1, conv_weights
    la t2, conv_biases
    la t3, conv_output   
    
    li t4, 8             
    vsetvli t5, t4, e8   
    
    li t6, 0            
conv_row_loop:
    li t4, 0            
conv_col_loop:
    vle8.v v0, (t2)      
    
    li s0, 0           
filter_row_loop:
    li s1, 0            
filter_col_loop:
    add s2, t6, s0       
    add s3, t4, s1       
    
    mul s4, s2, a1      
    add s4, s4, s3      
    add s4, a0, s4       
    
    lbu s5, 0(s4)        
    vmv.v.x v1, s5       
    
    mul s6, s0, a2       
    add s6, s6, s1      
    slli s6, s6, 3       
    add s6, t1, s6      
    
    vle8.v v2, (s6)
    
    vmul.vv v3, v1, v2
    vadd.vv v0, v0, v3
    
    addi s1, s1, 1
    blt s1, a2, filter_col_loop
    
    addi s0, s0, 1
    blt s0, a2, filter_row_loop
    
    vse8.v v0, (t3)
    addi t3, t3, 8      
    addi t4, t4, 1
    blt t4, t0, conv_col_loop
    
    addi t6, t6, 1
    blt t6, t0, conv_row_loop
    
    ret


relu:
    mul t0, a1, a2       
    mul t0, t0, a3    
    
    li t1, 16            
    vsetvli t2, t1, e8
    
    la t3, relu_output   
    li t4, 0             
    
relu_loop:
    vle8.v v0, (a0)
    
    vmax.vx v1, v0, zero
    
    vse8.v v1, (t3)
    
    add a0, a0, t2
    add t3, t3, t2
    add t4, t4, t2
    blt t4, t0, relu_loop
    
    ret


max_pool2d:
    div t0, a2, a4       
    
    li t1, 8
    vsetvli t2, t1, e8
    
    li t3, 0            
pool_row_loop:
    li t4, 0             
pool_col_loop:
    mul t5, t3, a4       
    mul t6, t4, a4       
    
    li t1, 0x80          
    vmv.v.x v0, t1       
    
    li s0, 0             
pool_inner_row:
    li s1, 0             
pool_inner_col:
    add s2, t5, s0       
    add s3, t6, s1      
    
    mul s4, s2, a2       
    add s4, s4, s3       
    mul s4, s4, a3       
    add s4, a0, s4      
    
    vle8.v v1, (s4)      
    
    vmax.vv v0, v0, v1
    
    addi s1, s1, 1
    blt s1, a4, pool_inner_col
    
    addi s0, s0, 1
    blt s0, a4, pool_inner_row
    
    vse8.v v0, (a1)
    addi a1, a1, 8       
    
    addi t4, t4, 1
    blt t4, t0, pool_col_loop
    
    addi t3, t3, 1
    blt t3, t0, pool_row_loop
    
    ret

flatten:
    mul t0, a2, a3       
    mul t0, t0, a4       
    
    mv t1, zero
flatten_loop:
    lb t2, 0(a0)
    sb t2, 0(a1)
    addi a0, a0, 1
    addi a1, a1, 1
    addi t1, t1, 1
    blt t1, t0, flatten_loop
    
    ret

dense_layer:
    mv t1, zero         
bias_copy_loop:
    lb t2, 0(a3)
    sb t2, 0(a1)
    addi a3, a3, 1
    addi a1, a1, 1
    addi t1, t1, 1
    blt t1, a5, bias_copy_loop    
   
    sub a1, a1, t1
     
    li t3, 0             
neuron_loop:
    mv t4, zero
    
    li t5, 0             
input_loop:
    lb t6, 0(a2)
    
    lb t0, 0(a0)
    
    mul t6, t6, t0
    add t4, t4, t6
    
    addi a0, a0, 1
    addi a2, a2, 1
    addi t5, t5, 1
    blt t5, a4, input_loop
    
    sub a0, a0, a4
    
    lb t6, 0(a1)
    add t6, t6, t4
    sb t6, 0(a1)
    addi a1, a1, 1
    addi t3, t3, 1
    blt t3, a5, neuron_loop
    
    ret

softmax:
    li t0, 0x80          
    mv t1, zero         
max_loop:
    lb t2, 0(a0)
    bge t0, t2, not_max
    mv t0, t2
not_max:
    addi a0, a0, 1
    addi t1, t1, 1
    blt t1, a2, max_loop
    sub a0, a0, t1      
    mv t3, zero          
    mv t1, zero         
sum_loop:
    lb t2, 0(a0)
    sub t2, t2, t0      
    li t4, 1            
    add t5, t4, t2       
    mul t6, t2, t2       
    srli t6, t6, 1       
    add t5, t5, t6       
    
    sb t5, 0(a1)
    add t3, t3, t5       
    
    addi a0, a0, 1
    addi a1, a1, 1
    addi t1, t1, 1
    blt t1, a2, sum_loop
    sub a1, a1, t1      
    mv t1, zero          
normalize_loop:
    lb t2, 0(a1)
    slli t2, t2, 8       
    div t2, t2, t3       
    sb t2, 0(a1)
    addi a1, a1, 1
    addi t1, t1, 1
    blt t1, a2, normalize_loop
    
    ret


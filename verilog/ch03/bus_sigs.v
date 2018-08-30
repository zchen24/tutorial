`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// 
// Module Name: bus_sigs
// Description: 
//     1) replication operator
//     2) bitwise negation operator 
// 
// Reference: 
//     Book: Verilog by Example P11
//////////////////////////////////////////////////////////////////////////////////


module bus_sigs(
    input [3:0] in_1,
    input [3:0] in_2,
    input in_3,
    output [3:0] out_1
    );
    
    wire[3:0] in_3_bus;
    
    assign in_3_bus = {4{in_3}};
    assign out_1 = (~in_3_bus & in_1) | (in_3_bus & in_2);
    
endmodule

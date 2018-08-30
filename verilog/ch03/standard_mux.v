`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// 
// Module Name: standard_mux
// Description: 
//     1) combinational conditional construct 
// 
// Reference: 
//     Book: Verilog by Example P13
//////////////////////////////////////////////////////////////////////////////////


module standard_mux(
    input [3:0] in_1,
    input [3:0] in_2,
    input in_3,
    output [3:0] out_1
    );
    
    assign out_1 = in_3 ? in_2 : in_1;
    
endmodule

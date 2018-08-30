`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// 
// Module Name: bus_breakout
// Description: 
//	1) bus concatenation 
//	2) MSB first (next to the left most brace), LSB last
// 
// Reference: 
//     Book: Verilog by Example P15
//////////////////////////////////////////////////////////////////////////////////


module bus_breakout(
    input [3:0] in_1,
    input [3:0] in_2,   
    output [5:0] out_1
    );
    
    assign out_1 = {
    	in_2[3:2],
    	in_2[1] & in_1[3],
    	in_2[0] & in_1[2],
    	in_1[1:0]
    };    
    
endmodule

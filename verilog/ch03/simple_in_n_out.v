`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// 
// Module Name: bus_breakout
// Description: 
//	1) how to define input/output ports
//  2) wire assign logic
// 
// Reference: 
//     Book: Verilog by Example P7
//////////////////////////////////////////////////////////////////////////////////


module simple_in_n_out(
    input in_1,
    input in_2,
    input in_3,
    input out_1,
    input out_2
    );
    
    assign out_1 = in_1 & in_2 & in_3;
    assign out_2 = in_1 | in_2 | in_3;    
endmodule

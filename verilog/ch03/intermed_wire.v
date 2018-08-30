`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// 
// Module Name: bus_sigs
// Description: 
//  1) How to declare and use intermediate signal (wire or reg)
// 
// Reference: 
//     Book: Verilog by Example P9
//////////////////////////////////////////////////////////////////////////////////


module intermed_wire(
    input in_1,
    input in_2,
    input in_3,
    output out_1,
    output out_2
    );
    
    // internal signals e.g. wire/reg
    wire intermediate_sig;
    
    assign intermediate_sig = in_1 & in_2;
    
    assign out_1 = intermediate_sig & in_3;
    assign out_2 = intermediate_sig | in_3;     
    
endmodule

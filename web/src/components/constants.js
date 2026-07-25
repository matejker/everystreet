import React from "react";

export const Spacer = ( { type } ) => {
    let size = "10px";
    switch (type){
        case "small":
            size = "5px";
            break;
        case "large":
            size = "20px";
            break;
        default:
            size = "10px";
     }

    return (<div style={ { margin: {size}, display: "block" } } />);
};

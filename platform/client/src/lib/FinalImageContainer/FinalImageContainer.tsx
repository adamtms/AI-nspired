import { ReactNode } from "react";
import {ImageContainerProps} from "../ImageContainer/ImageContainer";
import ImageContainer from "../ImageContainer";
import ComparisonPopup from "../ComparisonPopup";

import './FinalImageContainer.css'

export default class FinalImageContainer extends ImageContainer {
    popup: ComparisonPopup | null = null;
    constructor(props: ImageContainerProps) {
        props.source = "final";
        super(props);
        this.state = {
            ...this.state,
            selectedImage: ""
        };
    }

    handleImageClick(image: string) {
        this.setState({selectedImage: image});
        this.popup?.openPopup();
    }

    render(): ReactNode {
        return (
            <>
                <div className="image-container">
                    {this.state.images.map((image) => (
                        <div className="image">
                            <img className="clickable" src={"/images/" + image} alt="source" onClick={() => {this.handleImageClick(image)}}/>
                        </div>
                 
                 ))}
                </div>
                <ComparisonPopup ref={(cd) => { this.popup = cd; }} selectedGroup={this.props.selectedGroup} selectedImage={this.state.selectedImage ? this.state.selectedImage : ""}/>
            </>
        )
    }
}
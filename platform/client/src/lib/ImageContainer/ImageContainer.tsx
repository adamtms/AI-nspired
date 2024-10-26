import { Component, ReactNode } from "react";
import './ImageContainer.css'

export interface ImageContainerProps {
    selectedGroup: string;
    source: string;
}

export interface ImageContainerState {
    images: string[];
    selectedImage?: string;
}

export default class ImageContainer extends Component<ImageContainerProps, ImageContainerState> {
    constructor(props: ImageContainerProps) {
        super(props);
        this.state = {
            images: [] 
        };
    }

    async get_images() {
        let images:string[] = [];
        await fetch("/api/get-group-images", {
            method: "post",
            headers: {
                Accept: "application/json",
                "Content-Type": "application/json",
            },
            body: JSON.stringify({ group: this.props.selectedGroup }),
        })
            .then((res) => res.json())
            .then((data) => {
                images = data[this.props.source];
            });
        console.log(images);
        return images;
    }

    async componentDidMount() {
        const images = await this.get_images();
        this.setState({images: images});
    }

    async componentDidUpdate(prevProps: ImageContainerProps) {
        if (prevProps.selectedGroup !== this.props.selectedGroup) {
            console.log(`Group changed from ${prevProps.selectedGroup} to ${this.props.selectedGroup}`);
            const images = await this.get_images();
            this.setState({images: images});
        }
    }

    render(): ReactNode {
        return (
            <>
                <div className="image-container">
                    {this.state.images.sort().map((image) => (
                        <div className="image">
                            <img src={"/images/" + image} alt="source" />
                        </div>
                    ))}
                </div>
            </>
        )
    }
}

import { Component, ReactNode } from "react";
import "./ComparisonPopup.css";

interface ComparisonPopupProps {
  selectedGroup: string;
  selectedImage: string;
}

interface ComparisonPopupState {
  similarities: { image: string; similarity: number }[];
  popup_open: boolean;
  images_to_compare: string[];
}

export default class ComparisonPopup extends Component<
  ComparisonPopupProps,
  ComparisonPopupState
> {
  constructor(props: ComparisonPopupProps) {
    super(props);
    this.state = {
      similarities: [],
      popup_open: false,
      images_to_compare: [],
    };
  }

  openPopup() {
    this.setState({ popup_open: true });
  }

  closePopup() {
    this.setState({ popup_open: false });
  }

  async load_images_to_compare(): Promise<string[]> {
    return fetch("/api/get-group-images", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ group: this.props.selectedGroup }),
    })
      .then((response) => response.json())
      .then((data) => {
        const joined: string[] = [...data.web, ...data.ai];
        return joined;
      });
  }

  async load_similarities() {
    const similarities: { image: string; similarity: number }[] = [];
    for (const image of this.state.images_to_compare) {
      await fetch("/api/get-similarity", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ im1: this.props.selectedImage, im2: image }),
      })
        .then((response) => response.json())
        .then((data) => {
          similarities.push({ image: image, similarity: data.similarity });
          this.setState({ similarities: similarities }, () => {
            console.log(this.state.similarities.length);
          });
        });
    }
  }

  componentDidUpdate(prevProps: Readonly<ComparisonPopupProps>): void {
    if (prevProps.selectedImage !== this.props.selectedImage) {
      this.setState({ similarities: [], images_to_compare: [] });
      this.load_images_to_compare().then((images) => {
        this.setState({ images_to_compare: images }, this.load_similarities);
      });
    }
  }

  render(): ReactNode {
    if (!this.state.popup_open) {
      return null;
    }
    return (
      <>
        <div id="popup">
          <button
            onClick={() => {
              this.closePopup();
            }}
          >
            Close
          </button>
          <div className="split-horizontal">
            <div className="image-holder">
              <img src={"/images/" + this.props.selectedImage} alt="source" />
            </div>
            <div id="similar-images">
              <h2>Top 5 most similar images</h2>
              <div className="scroll-vertical">
                <div className="scroll-vertical">
                  {this.state.similarities
                    .sort((a, b) => b.similarity - a.similarity)
                    .slice(0, 5)
                    .map((similarity, index) => {
                      return (
                        <div key={index} className="similarity-image">
                          <img
                            src={"/images/" + similarity.image}
                            alt="similar"
                          />
                          <p>Similarity: {similarity.similarity}</p>
                        </div>
                      );
                    })}
                </div>
              </div>
            </div>
          </div>
        </div>
      </>
    );
  }
}

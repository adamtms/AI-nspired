// eslint-disable-next-line @typescript-eslint/no-explicit-any
export async function load_images_to_compare(selectedGroup: string): Promise<string[]> {
    return fetch(`${process.env.REACT_APP_BACKEND_URL}/api/get-group-images`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ group: selectedGroup }),
    })
      .then((response) => response.json())
      .then((data) => {
        const joined: string[] = [...data.web, ...data.ai];
        return joined;
      });
  }

export async function get_images(selectedGroup: string, source: string): Promise<string[]> {
    let images:string[] = [];
    await fetch(`${process.env.REACT_APP_BACKEND_URL}/api/get-group-images`, {
        method: "post",
        headers: {
            Accept: "application/json",
            "Content-Type": "application/json",
        },
        body: JSON.stringify({ group: selectedGroup }),
    })
        .then((res) => res.json())
        .then((data) => {
            images = data[source];
        });
    console.log(images);
    return images;
}
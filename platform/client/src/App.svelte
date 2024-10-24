<script>
    import { onMount } from "svelte";
    import ImageContainer from "./lib/ImageContainer.svelte";
    import SpecialImageContainer from "./lib/SpecialImageContainer.svelte";

    let groups_available = [];

    //group selection dropdown
    let group_selected = "8";

    onMount(() => {
        fetch("/api/get-groups")
            .then((res) => res.json())
            .then((data) => {
                groups_available = data.groups;
                groups_available.sort();
            });
    });
</script>

<main>
    <div id="header">
        <select bind:value={group_selected}>
            {#each groups_available as group}
                <option value={group}>{group}</option>
            {/each}
        </select>
    </div>
    <div id="content">
        <h1>Final</h1>
        <SpecialImageContainer group={group_selected} />
        <h1>Web</h1>
        <ImageContainer group={group_selected} source="web" />
        <h1>AI</h1>
        <ImageContainer group={group_selected} source="ai" />
    </div>
</main>

<style>
    main {
        display: flex;
        flex-direction: column;
        align-items: center;
        margin: 0;
        padding: 0;
        position: absolute;
        top: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
        max-width: 100vw;
    }
    #header {
        display: flex;
        flex-direction: row;
        justify-content: flex-start;
        margin: 1rem;
        width: 100vw;
    }
    #content {
        display: flex;
        flex-direction: column;
        align-items: center;
        flex-grow: 1;
        max-width: 90vw;
    }
</style>

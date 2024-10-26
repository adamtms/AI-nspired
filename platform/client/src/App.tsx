import { useState, useEffect } from "react";
import './App.css'
import ImageContainer from "./lib/ImageContainer";
import FinalImageContainer from "./lib/FinalImageContainer";

async function fetchGroups() {
  let groups_available: string[] = [];
  await fetch("/api/get-groups")
          .then((res) => res.json())
          .then((data) => {
              groups_available = data.groups;
              groups_available.sort((a, b) => parseInt(a) - parseInt(b));
          });
  return groups_available;
}

function App() {
  const [groups, setGroups] = useState<string[]>([]);
  const [selectedGroup, setSelectedGroup] = useState<string>('8');

  function handleGroupChange(event: React.ChangeEvent<HTMLSelectElement>) {
    setSelectedGroup(event.target.value);
    console.log(event.target.value);
  }

  useEffect(() => {
    fetchGroups().then((groups) => {setGroups(groups);setSelectedGroup(groups[0])});
  }, []);

  return (
    <>
      <div id="header">
        <select onChange={handleGroupChange}>
          {groups.map((group) => (
            <option value={group}>
              {group}
            </option>
          ))}
        </select>
        <h1>Group {selectedGroup}</h1>
      </div>
      <div id="content">
        <FinalImageContainer selectedGroup={selectedGroup} source="final"/>
        <ImageContainer selectedGroup={selectedGroup} source="web" />
        <ImageContainer selectedGroup={selectedGroup} source="ai" />
      </div>
    </>
  )
}

export default App

import React from "react";
import { createRoot } from "react-dom/client";
import "./profile.css";
import ProfilePage from "./ProfilePage";

const root = createRoot(document.getElementById("root")!);
root.render(<ProfilePage />);

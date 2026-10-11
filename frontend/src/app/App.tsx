import { createHashRouter, Navigate, RouterProvider } from "react-router-dom";
import { AppShell } from "@/app/AppShell";
import { AboutPage } from "@/features/about/AboutPage";
import { CalibrationPage } from "@/features/calibration/CalibrationPage";
import { RacePage } from "@/features/race/RacePage";
import { SeasonPage } from "@/features/season/SeasonPage";

// Hash routing works on GitHub Pages without server rewrites.
const router = createHashRouter([
  {
    element: <AppShell />,
    children: [
      { index: true, element: <Navigate to="/race" replace /> },
      { path: "race", element: <RacePage /> },
      { path: "season", element: <SeasonPage /> },
      { path: "calibration", element: <CalibrationPage /> },
      { path: "about", element: <AboutPage /> },
    ],
  },
]);

export function App() {
  return <RouterProvider router={router} />;
}

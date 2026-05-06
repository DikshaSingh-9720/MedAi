import { Navigate, Route, Routes } from "react-router-dom";
import Dashboard from "./pages/DashboardProfessional.jsx";
import Layout from "./components/Layout.jsx";

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<Layout />}>
        <Route index element={<Dashboard />} />
      </Route>
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}

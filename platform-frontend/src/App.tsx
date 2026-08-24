import { HashRouter, Route, Routes } from 'react-router-dom'
import Layout from './components/Layout'
import BattlePage from './pages/BattlePage'
import BenchmarkPage from './pages/BenchmarkPage'
import HistoryPage from './pages/HistoryPage'
import LearningPage from './pages/LearningPage'
import LobbyPage from './pages/LobbyPage'
import ReplayPage from './pages/ReplayPage'

export default function App() {
  return (
    <HashRouter>
      <Routes>
        <Route element={<Layout />}>
          <Route path="/" element={<LobbyPage />} />
          <Route path="/battle/:gameId" element={<BattlePage />} />
          <Route path="/benchmark" element={<BenchmarkPage />} />
          <Route path="/learning" element={<LearningPage />} />
          <Route path="/history" element={<HistoryPage />} />
          <Route path="/replay/:matchId" element={<ReplayPage />} />
        </Route>
      </Routes>
    </HashRouter>
  )
}

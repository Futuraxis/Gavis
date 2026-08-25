import { HashRouter, Route, Routes } from 'react-router-dom'
import Layout from './components/Layout'
import BattlePage from './pages/BattlePage'
import BenchmarkPage from './pages/BenchmarkPage'
import HistoryPage from './pages/HistoryPage'
import HomePage from './pages/HomePage'
import LearningPage from './pages/LearningPage'
import LobbyPage from './pages/LobbyPage'
import ProfilePage from './pages/ProfilePage'
import ReviewPage from './pages/ReviewPage'
import SettingsPage from './pages/SettingsPage'

export default function App() {
  return (
    <HashRouter>
      <Routes>
        <Route element={<Layout />}>
          <Route path="/" element={<HomePage />} />
          <Route path="/lobby" element={<LobbyPage />} />
          <Route path="/battle/:gameId" element={<BattlePage />} />
          <Route path="/benchmark" element={<BenchmarkPage />} />
          <Route path="/learning" element={<LearningPage />} />
          <Route path="/history" element={<HistoryPage />} />
          <Route path="/profile" element={<ProfilePage />} />
          <Route path="/settings" element={<SettingsPage />} />
          <Route path="/review/:matchId" element={<ReviewPage />} />
          <Route path="/replay/:matchId" element={<ReviewPage />} />
        </Route>
      </Routes>
    </HashRouter>
  )
}

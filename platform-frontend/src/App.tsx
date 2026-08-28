import { useEffect, useState } from 'react'
import { HashRouter, Route, Routes } from 'react-router-dom'
import ChatPage from './chat/ChatPage'
import Layout from './components/Layout'
import BattlePage from './pages/BattlePage'
import CreateGamePage from './pages/CreateGamePage'
import BenchmarkPage from './pages/BenchmarkPage'
import HistoryPage from './pages/HistoryPage'
import HomePage from './pages/HomePage'
import LearningPage from './pages/LearningPage'
import LobbyPage from './pages/LobbyPage'
import ProfilePage from './pages/ProfilePage'
import ReviewPage from './pages/ReviewPage'
import SettingsPage from './pages/SettingsPage'
import { PLATFORM_EVENT, VIEW_EVENT, loadChatStore } from './chat/sessionStore'

export default function App() {
  const [viewMode, setViewMode] = useState<'chat' | 'platform'>(() => loadChatStore().viewMode)

  // 全局切模式事件：ChatPage 头部按钮 → platform；Layout「回到对话」→ chat。
  useEffect(() => {
    const toChat = () => setViewMode('chat')
    const toPlatform = () => setViewMode('platform')
    window.addEventListener(VIEW_EVENT, toChat)
    window.addEventListener(PLATFORM_EVENT, toPlatform)
    return () => {
      window.removeEventListener(VIEW_EVENT, toChat)
      window.removeEventListener(PLATFORM_EVENT, toPlatform)
    }
  }, [])

  if (viewMode === 'chat') {
    return <ChatPage />
  }

  return (
    <HashRouter>
      <Routes>
        <Route element={<Layout />}>
          <Route path="/" element={<HomePage />} />
          <Route path="/lobby" element={<LobbyPage />} />
          <Route path="/create" element={<CreateGamePage />} />
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
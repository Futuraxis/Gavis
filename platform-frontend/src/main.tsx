import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App'
import { applyStoredTheme } from './settings'
import './chat/chat.css'
import './styles/global.css'

applyStoredTheme()

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
)

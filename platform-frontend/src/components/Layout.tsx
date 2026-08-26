import { NavLink, Outlet } from 'react-router-dom'

const NAV = [
  { to: '/', label: '首页', icon: '🏠', end: true },
  { to: '/lobby', label: '游戏大厅', icon: '🏛️', end: false },
  { to: '/create', label: '创建游戏', icon: '✨', end: false },
  { to: '/battle/moon_chess', label: '对战中心', icon: '⚔️', end: false },
  { to: '/history', label: '对局记录', icon: '📜', end: false },
  { to: '/profile', label: '个人中心', icon: '👤', end: false },
  { to: '/settings', label: '设置', icon: '⚙️', end: false },
  { to: '/benchmark', label: '求解器评测', icon: '📊', end: false },
  { to: '/learning', label: '在线学习', icon: '🧠', end: false },
]

export default function Layout() {
  return (
    <div className="layout">
      <aside className="sidebar">
        <div className="brand">🌙 Gavis 平台</div>
        <nav>
          {NAV.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.end}
              className={({ isActive }) => (isActive ? 'nav-item active' : 'nav-item')}
            >
              <span>{item.icon}</span>
              {item.label}
            </NavLink>
          ))}
        </nav>
      </aside>
      <main className="content">
        <Outlet />
      </main>
    </div>
  )
}

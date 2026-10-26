import React from 'react';
import { 
  ShieldAlert, 
  Activity, 
  Search, 
  Image as ImageIcon, 
  Layers, 
  BarChart3, 
  Cpu, 
  CheckCircle2,
  ChevronLeft,
  ChevronRight
} from 'lucide-react';

interface SidebarProps {
  currentTab: string;
  setCurrentTab: (tab: string) => void;
  collapsed: boolean;
  setCollapsed: (collapsed: boolean) => void;
  queueCount?: number;
}

export function Sidebar({ currentTab, setCurrentTab, collapsed, setCollapsed, queueCount = 0 }: SidebarProps) {
  const navItems = [
    { id: 'overview', label: 'Command Center', icon: Activity, badge: null },
    { id: 'analyze', label: 'Content Analyzer', icon: Search, badge: null },
    { id: 'meme', label: 'Meme Intelligence', icon: ImageIcon, badge: 'RESEARCH' },
    { id: 'queue', label: 'Moderation Queue', icon: Layers, badge: queueCount > 0 ? String(queueCount) : null },
    { id: 'analytics', label: 'Analytics Lab', icon: BarChart3, badge: null },
    { id: 'models', label: 'Model Intelligence', icon: Cpu, badge: null },
  ];

  return (
    <aside 
      className={`fixed top-0 left-0 h-screen z-40 transition-all duration-300 ease-in-out glass-panel border-r border-slate-800/80 flex flex-col justify-between ${
        collapsed ? 'w-20' : 'w-64'
      }`}
    >
      {/* Top Header & Branding */}
      <div>
        <div className="h-18 px-5 py-5 flex items-center justify-between border-b border-slate-800/60">
          {!collapsed ? (
            <div className="flex items-center space-x-3 overflow-hidden">
              <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-cyan-500 to-blue-600 flex items-center justify-center shadow-glow-cyan/50 flex-shrink-0">
                <ShieldAlert className="w-5 h-5 text-dark-950 font-bold" />
              </div>
              <div className="flex flex-col truncate">
                <span className="font-mono text-base font-extrabold tracking-wider text-slate-100 flex items-center gap-1.5">
                  TOXISCOPE
                  <span className="w-1.5 h-1.5 rounded-full bg-cyan-400"></span>
                </span>
                <span className="text-[10px] text-cyan-400/80 tracking-widest font-mono uppercase truncate">
                  Multilingual AI Moderation
                </span>
              </div>
            </div>
          ) : (
            <div className="mx-auto">
              <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-cyan-500 to-blue-600 flex items-center justify-center shadow-glow-cyan/50">
                <ShieldAlert className="w-5 h-5 text-dark-950 font-bold" />
              </div>
            </div>
          )}

          <button
            onClick={() => setCollapsed(!collapsed)}
            className="p-1 rounded-lg text-slate-400 hover:text-slate-200 hover:bg-slate-800/60 transition hidden md:block"
            title={collapsed ? "Expand sidebar" : "Collapse sidebar"}
          >
            {collapsed ? <ChevronRight className="w-4 h-4" /> : <ChevronLeft className="w-4 h-4" />}
          </button>
        </div>

        {/* Navigation Items */}
        <nav className="p-3 space-y-1.5">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = currentTab === item.id;
            return (
              <button
                key={item.id}
                onClick={() => setCurrentTab(item.id)}
                className={`w-full flex items-center rounded-xl px-3.5 py-3 transition-all duration-200 text-sm font-medium group relative ${
                  isActive
                    ? 'bg-gradient-to-r from-cyan-950/60 to-blue-950/40 text-cyan-300 border border-cyan-500/30 shadow-glow-cyan/20'
                    : 'text-slate-400 hover:text-slate-100 hover:bg-slate-800/40'
                }`}
                title={collapsed ? item.label : undefined}
              >
                <Icon className={`w-5 h-5 flex-shrink-0 transition-transform duration-200 ${
                  isActive ? 'text-cyan-400 scale-105' : 'text-slate-400 group-hover:text-slate-200'
                } ${collapsed ? 'mx-auto' : 'mr-3.5'}`} />

                {!collapsed && (
                  <span className="truncate flex-1 text-left">{item.label}</span>
                )}

                {!collapsed && item.badge && (
                  <span className={`text-[10px] font-mono px-2 py-0.5 rounded-full uppercase tracking-wider ${
                    item.badge === 'RESEARCH' 
                      ? 'bg-purple-950 text-purple-300 border border-purple-500/30'
                      : 'bg-rose-950 text-rose-300 border border-rose-500/30 font-bold'
                  }`}>
                    {item.badge}
                  </span>
                )}

                {isActive && (
                  <div className="absolute left-0 top-1/2 -translate-y-1/2 w-1 h-6 bg-cyan-400 rounded-r-full shadow-glow-cyan"></div>
                )}
              </button>
            );
          })}
        </nav>
      </div>

      {/* System Telemetry & Status Indicator */}
      <div className="p-3 border-t border-slate-800/60">
        {!collapsed ? (
          <div className="p-3 rounded-xl bg-dark-900/80 border border-slate-800/80 text-xs font-mono space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-slate-400 flex items-center gap-1.5">
                <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
                SYSTEM ONLINE
              </span>
              <span className="text-emerald-400 text-[11px]">ACTIVE</span>
            </div>
            <div className="text-[11px] text-slate-400 space-y-0.5 pt-1 border-t border-slate-800/50">
              <div className="flex justify-between">
                <span>MuRIL (Text)</span>
                <span className="text-cyan-400">17 Indic + En</span>
              </div>
              <div className="flex justify-between">
                <span>CLIP ViT-B/32</span>
                <span className="text-purple-400">Multimodal</span>
              </div>
              <div className="flex justify-between">
                <span>JEV Adjudicator</span>
                <span className="text-amber-400">Contextual</span>
              </div>
            </div>
          </div>
        ) : (
          <div className="flex justify-center p-2" title="System Online">
            <span className="w-3 h-3 rounded-full bg-emerald-400 animate-pulse shadow-glow-emerald"></span>
          </div>
        )}
      </div>
    </aside>
  );
}
export default Sidebar;

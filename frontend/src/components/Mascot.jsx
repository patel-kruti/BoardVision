import React from "react";

export default function Mascot() {
  return (
    <div className="hidden md:block fixed bottom-6 right-6 select-none">
      <div className="flex items-end gap-3">
        <div className="rounded-2xl border border-slate-200 bg-white/90 shadow-sm px-4 py-3 backdrop-blur">
          <div className="text-sm font-semibold text-slate-900">BoardBuddy</div>
          <div className="text-xs text-slate-600">
            Upload a flowchart photo, edit nodes/edges, then generate PPT.
          </div>
        </div>
        <svg
          width="74"
          height="74"
          viewBox="0 0 74 74"
          xmlns="http://www.w3.org/2000/svg"
          className="drop-shadow-sm"
        >
          <defs>
            <linearGradient id="g" x1="0" y1="0" x2="1" y2="1">
              <stop offset="0" stopColor="#60a5fa" />
              <stop offset="1" stopColor="#34d399" />
            </linearGradient>
          </defs>
          <circle cx="37" cy="37" r="34" fill="url(#g)" />
          <circle cx="27" cy="32" r="6" fill="#fff" />
          <circle cx="47" cy="32" r="6" fill="#fff" />
          <circle cx="29" cy="34" r="2.4" fill="#0f172a" />
          <circle cx="49" cy="34" r="2.4" fill="#0f172a" />
          <path
            d="M24 48c4 5 9 7 13 7s9-2 13-7"
            fill="none"
            stroke="#0f172a"
            strokeWidth="3"
            strokeLinecap="round"
          />
          <path
            d="M20 16c6-5 12-7 17-7s11 2 17 7"
            fill="none"
            stroke="rgba(255,255,255,.75)"
            strokeWidth="3"
            strokeLinecap="round"
          />
        </svg>
      </div>
    </div>
  );
}



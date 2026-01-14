'use client';

import React from 'react';

interface LanguageSelectorProps {
  selectedLanguage: string;
  onLanguageChange: (language: string) => void;
}

const languages = [
  { code: 'en_US', name: 'English', flag: '🇺🇸' },
  { code: 'es_ES', name: 'Spanish', flag: '🇪🇸' },
  { code: 'fr_FR', name: 'French', flag: '🇫🇷' }
];

export default function LanguageSelector({ selectedLanguage, onLanguageChange }: LanguageSelectorProps) {
  return (
    <div className="p-4 bg-white rounded-lg shadow-md">
      <label className="block text-sm font-medium text-gray-700 mb-2">
        Select Language
      </label>
      <div className="flex gap-2">
        {languages.map((lang) => (
          <button
            key={lang.code}
            onClick={() => onLanguageChange(lang.code)}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-all ${
              selectedLanguage === lang.code
                ? 'bg-blue-500 text-white shadow-md'
                : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
            }`}
          >
            <span className="text-xl">{lang.flag}</span>
            <span className="font-medium">{lang.name}</span>
          </button>
        ))}
      </div>
    </div>
  );
}

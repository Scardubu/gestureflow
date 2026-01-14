'use client';

interface LanguageSelectorProps {
  selected: string;
  onChange: (language: string) => void;
}

export default function LanguageSelector({ selected, onChange }: LanguageSelectorProps) {
  const languages = [
    { code: 'en', name: 'English' },
    { code: 'es', name: 'Español' },
    { code: 'fr', name: 'Français' },
  ];

  return (
    <div className="language-selector">
      {languages.map((lang) => (
        <button
          key={lang.code}
          className={`language-button ${selected === lang.code ? 'active' : ''}`}
          onClick={() => onChange(lang.code)}
        >
          {lang.name}
        </button>
      ))}
    </div>
  );
}

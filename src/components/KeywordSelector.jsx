import React from "react";
import Select from "react-select";

export default function KeywordSelector({ options, onChange }) {
    const selectOptions = options.map((kw) => ({ value: kw, label: kw }));

    return (
        <Select
            isMulti
            options={selectOptions}
            onChange={(selected) => onChange(selected.map((s) => s.value))}
            placeholder="e.g., cholesterol, syphilis, vitamin d..."
            styles={{
                control: (base) => ({ ...base, borderRadius: 12, padding: "4px" }),
            }}
        />
    );
}

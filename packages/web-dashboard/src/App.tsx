import React from 'react';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import { CssBaseline, Box, Paper, Typography } from '@mui/material';

const theme = createTheme({
  palette: {
    mode: 'dark',
  },
});

function App() {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Box sx={{ minHeight: '100vh', p: 3, background: '#0f172a' }}>
        <Paper sx={{ p: 4, textAlign: 'center' }}>
          <Typography variant="h3" gutterBottom>
            🚀 AI Model Sentinel
          </Typography>
          <Typography variant="h6">
            Enterprise Dashboard - Under Construction
          </Typography>
        </Paper>
      </Box>
    </ThemeProvider>
  );
}

export default App;